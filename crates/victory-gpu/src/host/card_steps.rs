//! Steps of a submission that read title memory flights in the air may
//! still write, done on the card in the flights' order instead of on the
//! CPU after a wait for them (`VICTORY_CARD_STEPS`, see [`stage`]).

use std::sync::Arc;

use super::*;
use crate::shadow::Shadow;

#[cfg(feature = "diagnostics")]
mod redirect_stats;

/// The stage (`VICTORY_CARD_STEPS`; off by default): `1` constant buffers
/// of descriptor tables (`Step::Fill` with a slot of its own), `2` the
/// title's buffer copies (`Step::Copy`) too, with the steps of the same
/// submission that depend on them. Only with the relay's second stage
/// (`relay_exact`).
///
/// Why: eleven dispatches a frame (the skinning passes, in two early
/// submissions) write through a view of the title's whole 174 MB heap, at
/// places they read out of job records in buffers: no bound of what they
/// write is known before they ran (frame capture replay2: 17af15c4015cb4b6,
/// 55679a0547649d0b, 89d08e561897d4e5, fdd56d97a806bcca and three more,
/// all through one view at offset 5920 of the title's bindless table). The
/// title's ring of constants lies in the same heap. One constant buffer of
/// 480 bytes and one copy of 20 KiB a frame, at a new place of the ring
/// each time, waited for these passes to land: 25-30 ms of a 105 ms frame,
/// the submitting thread standing (run record-1; with `VICTORY_FILL_TITLE=1`
/// in run relaycopy-1 the copy alone took over the wait).
///
/// What is done: a list of ours goes to the card in front of the
/// submission's lists, in the same call. It copies the bytes read from the
/// host copy that the flights write into a buffer of ours, and from there
/// to the destination's host copy (a copy), or the fill's descriptor slot
/// is pointed at that buffer (a fill; the slot is the fill's alone, the
/// root signatures say such descriptors are volatile, and the list that
/// names it has not gone to the card).
///
/// Why that is the value the CPU would have read after the wait. After the
/// wait title memory holds, for every byte read, what the title wrote, or
/// what the last of the flights that wrote it wrote (the landings put it
/// there, in order). The host copy on the card holds the same when our list
/// runs: every flight that may write one of these bytes writes this same
/// host copy ([`holder`]: a write through another host copy, a read-back
/// or a value written behind a flight there, and the old wait is taken),
/// they are all ahead of our list on the queue, and the title's own bytes
/// are brought up to date by a refresh step of our list, which is prepared
/// like any other list's (also under flights' declared writes:
/// `stale_under_flights`, which settles if it cannot). Nothing of this
/// submission writes these bytes earlier than our list: the steps are done
/// in front of all its lists in either case, and among themselves in the
/// order they stand in wherever one depends on another: [`plan`] takes
/// every copy and fill of the submission that writes what a taken one
/// reads or writes, or reads what a taken one writes. Every draw that
/// reads a buffer view of the heap relies on the same refresh and the same
/// order; what is new is only that a constant buffer and a copy do too.
/// Where a page is not mapped the host copy may hold other bytes than the
/// CPU's copy would have left: refused.
///
/// For a copy's destination our list declares a shader write, exactly the
/// bytes copied, after a refresh of them: from there on it is what any
/// shader's write is. The relay compares, and puts what changed into title
/// memory before the title is told anything of this flight (its
/// bottom-of-pipe writes and fences come after, in order); CPU steps of
/// later submissions that read it find it declared and wait, or are done
/// here. Refused: a step of the submission that cannot be taken (a texture,
/// vertices expanded, a fill of memory of another kind) reads the
/// destination behind the copy, or a refresh brings another host copy of it
/// up to date ([`plan`], [`elsewhere`]). While such a copy is in the air a
/// submission that refreshes another host copy of its destination, or
/// reads it in a step that is not checked otherwise, waits for it
/// ([`copied_read`]); with the copy done on the CPU it was there at once.
///
/// What stage 2 does not see: code of ours that reads title memory away
/// from a submission's steps (the pass over textures picked by index, while
/// a list is recorded). With the CPU's copy such a read of the destination
/// found the bytes at once; now it finds them when the flight has landed.
/// The landing tells the guards (`crate::watch::landing`), so a texture
/// made from older bytes is made again. This is why stage 2 is its own.
pub(super) fn stage() -> u32 {
    static STAGE: OnceLock<u32> = OnceLock::new();
    *STAGE.get_or_init(|| match std::env::var("VICTORY_CARD_STEPS").ok().as_deref() {
        Some("1") => 1,
        Some("2") => 2,
        _ => 0,
    })
}

/// Whether a copy at its place in a list ([`place`]) is only in the way of
/// what is done on the card in front of the lists if that stands behind it in
/// them (`VICTORY_PLACED_ORDER=1`; off: wherever it stands), and a
/// submission where it does is cut behind the copy's list ([`cut_or_plan`]). Off:
/// the case did not come up once in runs submit-1 to submit-3, so it has only
/// its tests behind it.
///
/// The fills of constant buffers read the title's ring (7,500 a frame in the
/// opening), and one copy of a few hundred bytes a frame writes into it at its
/// place in a list. The whole submission was refused for it, and stood 40 to
/// 60 ms until the flights had landed (run n19).
fn placed_order() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_PLACED_ORDER") {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_PLACED_ORDER"))
}

/// Enable read-through with `VICTORY_PLACED_FILL_REDIRECT=1`.
/// Diagnostic A/B runs can switch this without changing list recording.
fn redirect_fills() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_PLACED_FILL_REDIRECT") {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PLACED_FILL_REDIRECT").is_some_and(|v| v == "1"))
}

/// Why [`plan`] refuses for [`place`].
const BEHIND_PLACED: &str = "a copy at its place in a list writes what it reads, or reads or writes what it writes";
const ELIGIBLE_CPU_FILL: &str = "remaining_cpu_fill_eligible";

/// A planner observation stays local until this decision is actually used.
#[derive(Default)]
struct Behind {
    steps: Option<(usize, usize)>,
    cause: Option<&'static str>,
    retry_chosen: Vec<Planned>,
}

impl Behind {
    fn refuse(&mut self, steps: Option<(usize, usize)>, cause: &'static str) -> &'static str {
        self.steps = steps;
        self.cause = Some(cause);
        BEHIND_PLACED
    }
}

/// The bytes of our buffer between reading and writing: what one submission
/// may do this way (each place read once, however many fills take it).
const BETWEEN: u64 = 4 << 20;

/// The most steps of a submission done this way.
const MOST: usize = 12288;

/// How a fill's bytes get where its list reads them, if the card brings
/// them.
#[derive(Clone)]
pub(super) enum FillTo {
    /// A descriptor slot that shows the fill's memory, and nothing else does
    /// (title heap, slot): it is made to show our buffer instead.
    Slot(usize, u32),
    /// A buffer the card can write, filled from its beginning.
    Buffer(ID3D12Resource),
}

/// Title memory something in the air writes: the host copy it goes through
/// (`None`: none, the bytes reach title memory when the flight lands), from,
/// to, and whether it is the destination of a copy done here.
pub(super) struct Writer {
    shadow: Option<Arc<Shadow>>,
    from: usize,
    to: usize,
    copy: bool,
}

/// What `items` of the line write.
pub(super) fn writers(items: &[Late]) -> Vec<Writer> {
    let mut out = Vec::new();
    for item in items {
        match item {
            Late::Run(flight) => {
                for (shadow, ranges) in &flight.back {
                    out.extend(ranges.iter().map(|&(from, to)| Writer {
                        shadow: Some(shadow.clone()),
                        from,
                        to,
                        copy: false,
                    }));
                }
                out.extend(flight.texture_back_ranges.iter().map(|&(from, to)| Writer {
                    shadow: None,
                    from,
                    to,
                    copy: false,
                }));
                out.extend(flight.own.iter().flat_map(Taken::copied));
            }
            Late::Write(at, _, bytes) => out.push(Writer {
                shadow: None,
                from: *at,
                to: *at + *bytes,
                copy: false,
            }),
            Late::Signal(..) => {}
        }
    }
    out
}

/// The destinations of copies done here that have not landed.
pub(super) fn copies(line: &Line) -> Vec<Writer> {
    if stage() < 2 {
        return Vec::new();
    }
    let flying = line.line.iter().filter_map(|l| match l {
        Late::Run(flight) => flight.own.as_ref(),
        _ => None,
    });
    let landing = line.landing.iter().filter(|w| w.copy).map(|w| Writer {
        shadow: w.shadow.clone(),
        ..*w
    });
    flying.flat_map(Taken::copied).chain(landing).collect()
}

/// Whether `runs` refresh another host copy of a destination in `copies`,
/// or read it in a step that flights are not waited for otherwise
/// (`Step::Host`): that reads title memory, where the copy is not yet.
pub(super) fn copied_read(copies: &[Writer], runs: &[&HostList]) -> bool {
    if copies.is_empty() {
        return false;
    }
    runs.iter().any(|list| {
        list.deferred().copies.iter().any(|step| match step {
            Step::Refresh(shadow, at, len) => copies
                .iter()
                .any(|w| w.from < *at + *len && *at < w.to && w.shadow.as_ref().is_none_or(|s| !Arc::ptr_eq(s, shadow))),
            Step::Host(_, (from, to)) => copies.iter().any(|w| w.from < *to && *from < w.to),
            Step::Plane((at, len), _) => copies.iter().any(|w| w.from < at.saturating_add(*len) && *at < w.to),
            _ => false,
        })
    })
}

/// What a step of a submission does to title memory, as the plan sees it.
/// `asked`: flights in the air stop it from being done on the CPU now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Does {
    Copy { dst: usize, src: usize, len: usize, asked: bool },
    /// `to`: its memory can be given the bytes on the card.
    Fill { src: usize, len: usize, asked: bool, to: bool },
    /// Another read on the CPU (a texture, vertices expanded).
    Read { from: usize, to: usize, asked: bool },
    /// A refresh of a host copy (named by its address).
    Refresh { shadow: usize, from: usize, to: usize },
    /// A copy that its list does on the card ([`place`]).
    /// `from_cpu`: whether it copies from CPU upload memory rather than a
    /// source host copy on the card.
    Placed { dst: usize, src: usize, len: usize, from_cpu: bool },
    /// A shader dispatch/draw in the submission declares writes to `[from, to)`.
    GpuWrite { from: usize, to: usize },
    /// Anything, anywhere.
    Run,
}

/// One step chosen by [`plan`]: index in `does`, execution level, and an
/// optional redirected source address (for a fill reading through an earlier
/// card-sourced placed copy in the same submission).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Planned {
    index: usize,
    level: u32,
    redirect: Option<Redirect>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Redirect {
    address: usize,
    placed: usize,
}

fn meet(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// The bytes `d` reads and writes as a copy or a fill (empty ranges for
/// what it does not).
fn reads_writes(d: &Does) -> Option<((usize, usize), (usize, usize))> {
    match *d {
        Does::Copy { dst, src, len, .. } => Some(((src, src + len), (dst, dst + len))),
        Does::Fill { src, len, .. } => Some(((src, src + len), (0, 0))),
        _ => None,
    }
}

fn touched(ranges: &[(usize, usize)], r: (usize, usize)) -> bool {
    touches(ranges, r.0, r.1)
}

/// The steps of `does` (by place) to be done on the card, in order, each
/// with its level; or why not. All that are asked for, and every copy and
/// fill that is in the way of one that is done there: it writes what that
/// one reads or writes, or reads what it writes, whichever comes first. So
/// what is left on the CPU writes none of their bytes, and reads of them
/// only what a taken copy has not been told to write yet: bytes a step
/// ahead of it reads, in title memory as it is.
///
/// `behind`: for [`BEHIND_PLACED`], the copy at its place and the first step
/// behind it that is in its way.
///
/// Steps of a level are done together, every read before every write; a
/// step that writes what an earlier one reads or writes, or reads what it
/// writes, is of a higher level. Our list does them level by level.
fn plan(
    does: &[Does],
    stage: u32,
    by_order: bool,
    redirect_fills: bool,
    behind: &mut Behind,
) -> Result<Vec<Planned>, &'static str> {
    let takes = |d: &Does| match *d {
        Does::Copy { .. } if stage < 2 => Err("a buffer copy (VICTORY_CARD_STEPS=2 does it on the card)"),
        Does::Fill { to: false, .. } => Err("a fill of memory the card cannot be given the bytes of"),
        _ => Ok(()),
    };
    let mut taken = vec![false; does.len()];
    let (mut read, mut written) = (Vec::new(), Vec::new());
    let mut more = Vec::new();
    for (i, d) in does.iter().enumerate() {
        match *d {
            Does::Run => return Err("a step of ours runs"),
            Does::Read { asked: true, .. } => return Err("a texture reads it"),
            Does::Copy { asked: true, .. } | Does::Fill { asked: true, .. } => more.push(i),
            _ => {}
        }
    }
    while !more.is_empty() {
        // (Joined once for the round: one by one the 2,800 fills of a frame
        // sorted the ranges each time, 16 ms of the submitting thread.)
        let (mut reads, mut writes) = (Vec::new(), Vec::new());
        for i in std::mem::take(&mut more) {
            takes(&does[i])?;
            taken[i] = true;
            if let Some((r, w)) = reads_writes(&does[i]) {
                if w.0 < w.1 && meet(r, w) {
                    return Err("a copy onto itself");
                }
                reads.push(r);
                writes.push(w);
            }
        }
        join_ranges(&mut read, reads);
        if writes.iter().any(|w| w.0 < w.1) {
            join_ranges(&mut written, writes);
        }
        if taken.iter().filter(|t| **t).count() > MOST {
            return Err("too many steps depend on it");
        }
        for (i, d) in does.iter().enumerate() {
            // A fill that cannot be taken is looked at below.
            if taken[i] || matches!(d, Does::Fill { to: false, .. }) {
                continue;
            }
            if let Some((r, w)) = reads_writes(d)
                && (touched(&written, r) || touched(&written, w) || touched(&read, w))
            {
                more.push(i);
            }
        }
    }
    // A copy at its place in a list comes after what is done in front of all
    // of them: as it should after the steps ahead of it, but not after those
    // behind it that read or write its bytes.
    //
    // Under `by_order` (`VICTORY_PLACED_ORDER=1`), a taken `Fill` behind a
    // card-sourced placed copy `P` whose source lies wholly inside `P`'s
    // destination can read `P`'s source host copy at the matching offset in
    // the front list instead of refusing the whole submission — provided no
    // earlier placed copy or copy anywhere in the submission writes `P`'s
    // source, and no intervening step overwrites the fill's bytes.
    let mut redirects = vec![None; does.len()];
    for (p, d) in does.iter().enumerate() {
        if let Does::Placed { dst, src, len, from_cpu } = *d
            && (touched(&read, (dst, dst + len)) || touched(&written, (dst, dst + len)) || touched(&written, (src, src + len)))
        {
            let (to, from) = ((dst, dst + len), (src, src + len));
            let in_the_way = |i: usize| {
                reads_writes(&does[i]).is_some_and(|(r, w)| meet(r, to) || w.0 < w.1 && (meet(w, to) || meet(w, from)))
            };
            if !by_order || !redirect_fills {
                let found = (p + 1..does.len()).find(|&i| taken[i] && in_the_way(i));
                if !by_order || found.is_some() {
                    let cause = if by_order { "redirect_disabled" } else { "ordering_disabled" };
                    return Err(behind.refuse(found.map(|i| (p, i)), cause));
                }
                behind.steps = None;
                continue;
            }
            let src_clean = || {
                !does.iter().enumerate().any(|(k, step)| match *step {
                    Does::Copy { dst: cd, len: cl, .. } => meet((cd, cd + cl), from),
                    Does::Placed { dst: pd, len: pl, .. } if k < p => meet((pd, pd + pl), from),
                    Does::GpuWrite { from: gf, to: gt } if k < p => meet((gf, gt), from),
                    _ => false,
                })
            };
            let mut src_checked: Option<bool> = None;
            // What stands between the copy and the step looked at, gathered
            // on the way: the later copies at their places, and what else
            // writes title memory there. (`place.rs` calls `title_written()`
            // for every `Step::Placed`, which clears `gpu_written`: a shader
            // write behind `p` is declared again as a `Step::GpuWrite` behind
            // it.)
            let mut later_placed: Vec<(usize, usize)> = Vec::new();
            let mut writes_between: Vec<(usize, usize)> = Vec::new();
            for r in p + 1..does.len() {
                if taken[r] && in_the_way(r) {
                    let Does::Fill { src: f_src, len: f_len, .. } = does[r] else {
                        return Err(behind.refuse(Some((p, r)), "consumer_is_copy"));
                    };
                    let f_range = (f_src, f_src + f_len);
                    // If a later placed copy before `r` also writes `f_range`,
                    // `p` is shadowed only when that later copy covers `f_range`
                    // completely (and will be validated when the loop reaches it).
                    if let Some(&(ld, le)) = later_placed.iter().rev().find(|&&later| meet(later, f_range)) {
                        if f_src >= ld && f_src + f_len <= le {
                            continue;
                        }
                        return Err(behind.refuse(Some((p, r)), "partial_newer_copy"));
                    }
                    let inside = f_src >= dst && f_src + f_len <= dst + len;
                    let dst_clean_between = !writes_between.iter().any(|&w| meet(w, f_range));
                    let cause = if from_cpu {
                        Some("upload_source")
                    } else if !inside {
                        Some("partial_coverage")
                    } else if !dst_clean_between {
                        Some("destination_overwritten")
                    } else if !*src_checked.get_or_insert_with(src_clean) {
                        Some("source_overwritten")
                    } else {
                        None
                    };
                    if let Some(cause) = cause {
                        return Err(behind.refuse(Some((p, r)), cause));
                    }
                    redirects[r] = Some(Redirect { address: src + (f_src - dst), placed: p });
                    continue;
                }
                match does[r] {
                    Does::Placed { dst: pd, len: pl, .. } => later_placed.push((pd, pd + pl)),
                    Does::Copy { dst: cd, len: cl, .. } => writes_between.push((cd, cd + cl)),
                    Does::GpuWrite { from: gf, to: gt } => writes_between.push((gf, gt)),
                    _ => {}
                }
            }
        }
    }
    let order: Vec<usize> = (0..does.len()).filter(|&i| taken[i]).collect();
    // What stays on the CPU and reads what an earlier taken copy writes
    // would read it too early.
    let mut behind: Vec<(usize, usize)> = Vec::new();
    for (i, d) in does.iter().enumerate() {
        if taken[i] {
            if let Some((_, w)) = reads_writes(d)
                && w.0 < w.1
            {
                join_ranges(&mut behind, [w]);
            }
            continue;
        }
        match *d {
            Does::Read { from, to, .. } if touched(&behind, (from, to)) => {
                return Err("a texture or expanded vertices read what it writes");
            }
            Does::Fill { src, len, .. } if touched(&behind, (src, src + len)) => {
                return Err("a fill of memory the card cannot be given the bytes of reads what it writes");
            }
            _ => {}
        }
    }
    let effective_rw = |i: usize| -> Option<((usize, usize), (usize, usize))> {
        let (r, w) = reads_writes(&does[i])?;
        let r = match redirects[i] {
            Some(Redirect { address, .. }) => (address, address + (r.1 - r.0)),
            None => r,
        };
        Some((r, w))
    };
    let mut out: Vec<Planned> = Vec::with_capacity(order.len());
    // The copies among them, which alone put others behind them or have to
    // stand behind others.
    let mut copies: Vec<usize> = Vec::new();
    for &i in &order {
        let Some((r, w)) = effective_rw(i) else {
            continue;
        };
        let writes = w.0 < w.1;
        let mut level = 0;
        // Behind what writes what this reads or writes, and, if this
        // writes, behind what reads it.
        let ahead: Box<dyn Iterator<Item = usize>> =
            if writes { Box::new(0..out.len()) } else { Box::new(copies.iter().copied()) };
        for k in ahead {
            let Planned { index: j, level: at, .. } = out[k];
            let Some((jr, jw)) = effective_rw(j) else {
                continue;
            };
            if at >= level && (meet(jw, r) || meet(jw, w) || meet(jr, w)) {
                level = at + 1;
            }
        }
        if writes {
            copies.push(out.len());
        }
        out.push(Planned {
            index: i,
            level,
            redirect: redirects[i],
        });
    }
    Ok(out)
}

/// A placed copy whose bytes still depend on a flight does not update
/// title memory during CPU preparation. Any CPU reader of its destination
/// must wait, even if another fill was redirected to the card.
/// A CPU upload source blocked directly by a flight cannot be promoted.
fn remaining_cpu_reader(
    does: &[Does],
    chosen: &[Planned],
    written: &[(usize, usize)],
    readers_before_copy: bool,
    mut stop: impl FnMut(usize) -> bool,
) -> Option<(usize, usize)> {
    let mut pending = Vec::new();
    for (i, step) in does.iter().enumerate() {
        if let Does::Placed { dst, src, len, from_cpu } = *step {
            let source_pending = touches(written, src, src + len);
            if from_cpu && source_pending {
                return Some((i, i));
            }
            if source_pending || touches(written, dst, dst + len) {
                pending.push((i, (dst, dst + len)));
            }
        }
    }
    if pending.is_empty() {
        return None;
    }
    // Planned entries follow their original order.
    let mut chosen = chosen.iter().peekable();
    for (i, step) in does.iter().enumerate() {
        if chosen.next_if(|p| p.index == i).is_some() {
            continue;
        }
        let read = match *step {
            Does::Copy { src, len, .. } | Does::Fill { src, len, .. }
            | Does::Placed { src, len, from_cpu: true, .. } => (src, src + len),
            Does::Read { from, to, .. } => (from, to),
            _ => continue,
        };
        if let Some(&(placed, _)) = pending.iter().find(|&&(placed, dst)| {
            (readers_before_copy || placed < i) && meet(read, dst)
        }) && stop(i) {
            return Some((placed, i));
        }
    }
    None
}

/// A fill has a GPU destination and the last placed copy touching it covers
/// the full range. The planner still checks intervening writes and the host
/// verifies the recorded backing and aliases before recording anything.
fn can_promote_fill(does: &[Does], step: usize) -> bool {
    let Does::Fill { src, len, to: true, .. } = does[step] else {
        return false;
    };
    if len == 0 {
        return false;
    }
    let read = (src, src + len);
    let previous = does[..step].iter().rev().find(|d| {
        matches!(**d, Does::Placed { dst, len, .. } if meet(read, (dst, dst + len)))
    });
    matches!(previous, Some(Does::Placed { dst, len, from_cpu: false, .. })
        if *dst <= read.0 && read.1 <= *dst + *len)
}

fn plan_with_flights(
    does: &[Does],
    written: &[(usize, usize)],
    stage: u32,
    by_order: bool,
    redirect_fills: bool,
    behind: &mut Behind,
) -> Result<Vec<Planned>, &'static str> {
    let chosen = plan(does, stage, by_order, redirect_fills, behind)?;
    if !chosen.is_empty()
        && let Some(reader) = remaining_cpu_reader(does, &chosen, written, place::readers_before_copy(), |_| true)
    {
        let (placed, step) = reader;
        let cause = if placed == step {
            "pending_upload_source"
        } else {
            match does[step] {
                Does::Copy { .. } => "remaining_cpu_copy",
                Does::Fill { to: false, .. } => "remaining_cpu_fill_cpu_only",
                Does::Fill { .. } if can_promote_fill(does, step) => ELIGIBLE_CPU_FILL,
                Does::Fill { .. } => "remaining_cpu_fill_without_covering_copy",
                Does::Placed { .. } => "remaining_cpu_placed_copy",
                _ => "remaining_cpu_read",
            }
        };
        if by_order && redirect_fills && cause == ELIGIBLE_CPU_FILL {
            behind.retry_chosen = chosen;
        }
        return Err(behind.refuse(Some(reader), cause));
    }
    Ok(chosen)
}

/// Re-plan once, keeping the original refusal if any CPU dependency or
/// placed-copy restriction remains. Called only by preparation, after the
/// original split decision has been used or ruled out.
fn retry_cpu_fills(
    does: &[Does],
    written: &[(usize, usize)],
    stage: u32,
    by_order: bool,
    redirect_fills: bool,
    original: &Behind,
) -> Option<(Vec<Planned>, u64)> {
    if !by_order || !redirect_fills || original.cause != Some(ELIGIBLE_CPU_FILL) {
        return None;
    }
    let mut promoted = Vec::new();
    let still_waits = remaining_cpu_reader(
        does, &original.retry_chosen, written, place::readers_before_copy(), |i| {
            if matches!(does[i], Does::Fill { asked: false, .. }) && can_promote_fill(does, i) {
                promoted.push(i);
                false
            } else {
                true
            }
        },
    );
    if still_waits.is_some() || promoted.is_empty() {
        return None;
    }
    let mut retry = does.to_vec();
    for &i in &promoted {
        if let Does::Fill { asked, .. } = &mut retry[i] {
            *asked = true;
        }
    }
    let chosen = plan_with_flights(
        &retry, written, stage, by_order, redirect_fills, &mut Behind::default(),
    ).ok()?;
    if !promoted.iter().all(|i| {
        chosen.binary_search_by_key(i, |p| p.index).is_ok_and(|at| chosen[at].redirect.is_some())
    }) {
        return None;
    }
    Some((chosen, promoted.len() as u64))
}

/// Whether a refresh of `does` brings a host copy other than `shadow` up to
/// date in `[from, to)`.
fn elsewhere(does: &[Does], shadow: usize, from: usize, to: usize) -> bool {
    does.iter().any(|d| match *d {
        Does::Refresh { shadow: s, from: a, to: b } => s != shadow && meet((a, b), (from, to)),
        _ => false,
    })
}

/// The host copy that holds `[from, to)` on the card once what `writers`
/// write is there: the one they all write it through, or without any the
/// one shaders would be given.
fn holder(writers: &[Writer], from: usize, to: usize) -> Result<Arc<Shadow>, &'static str> {
    let mut found: Option<&Arc<Shadow>> = None;
    for w in writers.iter().filter(|w| meet((w.from, w.to), (from, to))) {
        match (&w.shadow, found) {
            (None, _) => return Err("a read-back or a value behind a flight is written there"),
            (Some(s), Some(f)) if !Arc::ptr_eq(s, f) => return Err("flights write it through two host copies"),
            (Some(s), _) => found = Some(s),
        }
    }
    let shadow = match found {
        Some(s) => s.clone(),
        None => crate::shadow::find(from as u64, (to - from) as u64).ok_or("no host copy of it")?,
    };
    let (start, len) = shadow.span();
    if shadow.in_place() || (from as u64) < start || to as u64 > start + len {
        return Err("no host copy holds all of it");
    }
    Ok(shadow)
}

/// What the steps of `runs` do, and which step each is (its list by address,
/// its place in it). `written`: what flights may write.
fn does_of(written: &[(usize, usize)], runs: &[&HostList]) -> (Vec<Does>, Vec<(usize, usize)>) {
    let mut does = Vec::new();
    // Which step each is. One list's lock at a time: a submission may
    // name a list twice.
    let mut steps = Vec::new();
    for list in runs {
        for (i, step) in list.deferred().copies.iter().enumerate() {
            does.push(match step {
                &Step::Copy(dst, src, len) => Does::Copy {
                    dst,
                    src,
                    len,
                    asked: copy_blocked(written, dst, src, len).is_some(),
                },
                Step::Fill(_, src, len, to) => Does::Fill {
                    src: *src,
                    len: *len,
                    asked: read_blocked(written, *src, *len),
                    to: to.is_some(),
                },
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    Does::Read {
                        from: at,
                        to: at + len,
                        asked: touches(written, at, at + len),
                    }
                }
                &Step::Plane((at, len), _) => Does::Read {
                    from: at,
                    to: at.saturating_add(len),
                    asked: touches(written, at, at.saturating_add(len)),
                },
                &Step::Host(_, (from, to)) if from < to => Does::Read { from, to, asked: false },
                Step::Refresh(shadow, at, len) => Does::Refresh {
                    shadow: Arc::as_ptr(shadow) as usize,
                    from: *at,
                    to: *at + *len,
                },
                Step::RawCopy { .. } => Does::Run,
                Step::Placed(p) => Does::Placed { dst: p.dst, src: p.src, len: p.len, from_cpu: p.reads_upload() },
                &Step::GpuWrite(_, address, len) if len > 0 => Does::GpuWrite { from: address, to: address + len },
                Step::GpuWrite(..) | Step::Host(..) => continue,
            });
            steps.push((std::ptr::from_ref::<HostList>(list) as usize, i));
        }
    }
    (does, steps)
}

/// What [`cut_or_plan`] worked out for a submission it did not cut: the
/// steps and the plan of them, which [`Host::card_steps`] takes as they are
/// when the flights still write the same ranges (`written`) and the
/// title-memory fill heuristic is off. With the heuristic enabled, the
/// watch's answer can change while the flight ranges stay the same.
pub(super) struct Preplan {
    written: Vec<(usize, usize)>,
    does: Vec<Does>,
    steps: Vec<(usize, usize)>,
    redirect_fills: bool,
    chosen: Result<Vec<Planned>, &'static str>,
    behind: Behind,
}

pub(super) struct Cut {
    pub(super) at: usize,
    #[cfg(feature = "diagnostics")]
    trace: (bool, Option<&'static str>),
}

impl Cut {
    #[cfg(feature = "diagnostics")]
    pub(super) fn applied(&self) {
        PLACED_STATS[1].fetch_add(1, Ordering::Relaxed);
        let (on, cause) = self.trace;
        if let Some(cause) = cause {
            redirect_stats::cut(on, cause);
        }
    }
}

#[cfg(feature = "diagnostics")]
pub(super) struct RedirectWait {
    on: bool,
    why: &'static str,
}

#[cfg(feature = "diagnostics")]
impl RedirectWait {
    pub(super) fn finished(self, elapsed: std::time::Duration) {
        redirect_stats::wait(self.on, self.why, elapsed);
    }
}

/// What [`cut_or_plan`] found.
pub(super) enum CutOrPlan {
    /// The submission is cut before this one of its lists.
    Cut(Cut),
    /// It is not cut; its steps and their plan, for [`Host::card_steps`].
    Plan(Preplan),
    /// It is not cut, and nothing was worked out.
    None,
}

/// With flights in the air: before which of `runs` the submission is cut so
/// that what reads or writes the bytes of a copy at its place in an earlier
/// list ([`BEHIND_PLACED`]) is done on the card in front of a submission of
/// its own, behind that copy. A submission that is not cut keeps the steps
/// and the plan made to find that out ([`Preplan`]): they were made a second
/// time under the submission lock.
pub(super) fn cut_or_plan(runs: &[&HostList]) -> CutOrPlan {
    if stage() == 0 || !relay_exact() || !placed_order() || runs.len() < 2 {
        return CutOrPlan::None;
    }
    // (Most submissions have none.)
    let Some(first) = runs.iter().position(|l| l.deferred().copies.iter().any(|s| matches!(s, Step::Placed(_)))) else {
        return CutOrPlan::None;
    };
    if first + 1 == runs.len() {
        return CutOrPlan::None;
    }
    let written = line().written.clone();
    if blocked_by(&written, runs, false).is_none() {
        return CutOrPlan::None;
    }
    let (does, steps) = does_of(&written, runs);
    let mut behind = Behind::default();
    let redirect_fills = redirect_fills();
    let chosen = plan_with_flights(&does, &written, stage(), true, redirect_fills, &mut behind);
    if chosen == Err(BEHIND_PLACED) {
        let cut = behind.steps.and_then(|(placed, reader)| {
            let place = |i: usize| runs.iter().position(|l| std::ptr::from_ref::<HostList>(l) as usize == steps[i].0);
            let cut = place(placed)? + 1;
            (place(reader)? >= cut).then_some(cut)
        });
        if let Some(at) = cut {
            return CutOrPlan::Cut(Cut {
                at,
                #[cfg(feature = "diagnostics")]
                trace: (redirect_fills, behind.cause),
            });
        }
    }
    if fill_title_mode() != 0 {
        return CutOrPlan::None;
    }
    #[cfg(feature = "diagnostics")]
    if crate::ab::bit("VICTORY_FILL_TITLE").is_some() {
        return CutOrPlan::None;
    }
    CutOrPlan::Plan(Preplan { written, does, steps, redirect_fills, chosen, behind })
}

/// Whether every page of `[from, to)` can be read (through the kernel:
/// asking Windows walks the page tables, milliseconds in the title's heap).
/// `known`: pages that could be, of the same submission: its fills' places
/// lie eight to a page, and each was read whole to find out.
fn mapped(from: usize, to: usize, known: &mut HashMap<usize, ()>) -> bool {
    const PAGE: usize = 4096;
    (from / PAGE..to.div_ceil(PAGE)).all(|page| {
        known.contains_key(&page) || {
            let there = read_into((page * PAGE).max(from) as *const u8, &mut [0]);
            if there {
                known.insert(page, ());
            }
            there
        }
    })
}

/// A table by addresses and numbers of ours.
type HashMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<AddressHash>>;

/// Where the bytes of the fill that is `step` (its list by address, its place
/// in it) go.
fn fill_to(runs: &[&HostList], step: (usize, usize)) -> Option<FillTo> {
    let list = runs.iter().find(|l| std::ptr::from_ref::<HostList>(l) as usize == step.0)?;
    match list.deferred().copies.get(step.1)? {
        Step::Fill(_, _, _, to) => to.clone(),
        _ => None,
    }
}

fn require_recorded_source<T>(recorded: Option<&Arc<T>>, resolved: &Arc<T>) -> Result<(), &'static str> {
    if recorded.is_some_and(|source| Arc::ptr_eq(source, resolved)) {
        Ok(())
    } else {
        Err("a redirected fill would read a different host copy than the placed copy")
    }
}

fn verify_redirect(
    runs: &[&HostList],
    step: (usize, usize),
    resolved: &Arc<Shadow>,
) -> Result<(), &'static str> {
    let list = runs
        .iter()
        .find(|l| std::ptr::from_ref::<HostList>(l) as usize == step.0)
        .ok_or("the placed copy's recorded list is missing")?;
    let deferred = list.deferred();
    let Some(Step::Placed(placed)) = deferred.copies.get(step.1) else {
        return Err("the placed copy's recorded step is missing");
    };
    require_recorded_source(placed.source.as_ref(), resolved)?;
    let _mapping = crate::watch::title_memory_reading_now()
        .ok_or("title memory is being remapped while redirecting a fill")?;
    if crate::inplace::has_alias(placed.src, placed.src + placed.len)
        || crate::inplace::has_alias(placed.dst, placed.dst + placed.len)
    {
        return Err("a redirected fill depends on a placed copy of aliased title memory");
    }
    Ok(())
}

/// A step done on the card.
struct Op {
    from: (Arc<Shadow>, usize),
    len: usize,
    /// Its place in the buffer between.
    at: u64,
    level: u32,
    to: To,
}

enum To {
    /// A host copy, and the title address.
    Copy(Arc<Shadow>, usize),
    /// A descriptor slot (title heap, slot).
    Slot(usize, u32),
    /// A buffer's beginning.
    Buffer(ID3D12Resource),
}

/// Records a level of what [`stage`] describes into `list` (open): `reads`
/// are copied into `between` (source, offset in it, offset in `between`,
/// bytes), then `writes` out of it (destination, offset in it, offset in
/// `between`, bytes). `between` is left readable as constants (and must be
/// so, unless this is the `first` level), the others in the common state,
/// as the submission's lists expect them.
fn record(
    first: bool,
    list: &ID3D12GraphicsCommandList,
    between: &ID3D12Resource,
    reads: &[(&ID3D12Resource, u64, u64, u64)],
    writes: &[(&ID3D12Resource, u64, u64, u64)],
) {
    let common = D3D12_RESOURCE_STATE_COMMON;
    let (source, dest) = (D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    let mut sources: Vec<&ID3D12Resource> = Vec::new();
    let mut dests: Vec<&ID3D12Resource> = Vec::new();
    for (all, list) in [(reads, &mut sources), (writes, &mut dests)] {
        for &(buffer, ..) in all {
            if !list.iter().any(|b| b.as_raw() == buffer.as_raw()) {
                list.push(buffer);
            }
        }
    }
    let both = |b: &ID3D12Resource| dests.iter().any(|d| d.as_raw() == b.as_raw());
    let constants = source | D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    let mut barriers = vec![transition(between, dest, constants)];
    barriers.extend(sources.iter().map(|b| transition(b, source, if both(b) { dest } else { common })));
    let done: Vec<D3D12_RESOURCE_BARRIER> = dests.iter().map(|b| transition(b, dest, common)).collect();
    // SAFETY: an open list; live buffers, and the callers' ranges lie in
    // them. The buffers start in the common state (the list is the first of
    // its call) and are promoted by their first use.
    unsafe {
        if first {
            // What earlier lists' shaders wrote, first.
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(None),
                    }),
                },
            }]);
        } else {
            list.ResourceBarrier(&[transition(between, constants, dest)]);
        }
        for &(buffer, offset, at, len) in reads {
            list.CopyBufferRegion(between, at, buffer, offset, len);
        }
        list.ResourceBarrier(&barriers);
        for &(buffer, offset, at, len) in writes {
            list.CopyBufferRegion(buffer, offset, between, at, len);
        }
        if !done.is_empty() {
            list.ResourceBarrier(&done);
        }
    }
}

/// The list in front of a submission's, and its buffer ([`BETWEEN`] bytes).
struct Own {
    list: HostList,
    between: ID3D12Resource,
}

// SAFETY: D3D12 objects are free-threaded, and the list is touched by one
// thread at a time (the submitting one until it is sent, then the one that
// lands it, then nobody until it is taken out of `POOL`).
unsafe impl Send for Own {}

/// Those that have landed: closed, their steps dropped.
static POOL: Mutex<Vec<Own>> = Mutex::new(Vec::new());

/// Since the last frame-rate line: fills and copies done on the card, their
/// bytes, and submissions that waited instead, refused by a check.
static DONE: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Diagnostics: `[0]` fills redirected through a placed copy's source, `[1]`
/// submissions cut behind a placed copy (`cut_or_plan`).
#[cfg(feature = "diagnostics")]
static PLACED_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// What [`Host::card_steps`] took off a submission: the list that does it,
/// for the submission to run first, and the steps the CPU leaves out (their
/// list by address, their place in it).
pub(super) struct Taken {
    own: Own,
    steps: std::collections::HashSet<(usize, usize), std::hash::BuildHasherDefault<AddressHash>>,
    copied: Vec<(Arc<Shadow>, usize, usize)>,
}

impl Taken {
    pub(super) fn list(&self) -> &HostList {
        &self.own.list
    }

    /// Whether step `i` of `list` is done by this.
    pub(super) fn takes(&self, list: &HostList, i: usize) -> bool {
        self.steps.contains(&(std::ptr::from_ref(list) as usize, i))
    }

    fn copied(&self) -> impl Iterator<Item = Writer> {
        self.copied.iter().map(|(shadow, from, to)| Writer {
            shadow: Some(shadow.clone()),
            from: *from,
            to: *to,
            copy: true,
        })
    }

    /// The submission has run and landed: the list and its buffer are free.
    pub(super) fn landed(self) {
        // The host copies its steps held go; its commands stay until the
        // next use resets it.
        drop(std::mem::take(&mut *self.own.list.deferred()));
        POOL.lock().unwrap_or_else(|e| e.into_inner()).push(self.own);
    }
}

impl Host {
    /// With flights in the air: if what keeps `runs` from being prepared
    /// now is reads of title memory that can all be done on the card
    /// ([`stage`]), the list that does them. `None` if nothing is in the
    /// way or the old wait is due. Under the submission lock.
    pub(super) fn card_steps(
        &self,
        runs: &[&HostList],
        preplan: Option<Preplan>,
        #[cfg(feature = "diagnostics")] wait: &mut Option<RedirectWait>,
    ) -> Option<Taken> {
        if stage() == 0 || !relay_exact() {
            return None;
        }
        let (written, copies) = {
            let line = line();
            if line.written.is_empty() {
                return None;
            }
            (line.written.clone(), copies(&line))
        };
        let looked = crate::timeline::span("steps on the card: whether in the way");
        blocked_by(&written, runs, false)?;
        let read = copied_read(&copies, runs);
        looked.end();
        let taken = if read {
            Err("a copy done on the card is read through another host copy")
        } else {
            // The plan was made with placed-copy ordering enabled.
            let by_order = placed_order();
            let redirect_fills = redirect_fills();
            let preplan = preplan.filter(|p| p.written == written && by_order && p.redirect_fills == redirect_fills);
            #[cfg(feature = "diagnostics")]
            let mut trace = redirect_stats::Attempt::new(redirect_fills);
            let result = self.take_steps(
                &written, runs, preplan, by_order, redirect_fills,
                #[cfg(feature = "diagnostics")]
                &mut trace,
            );
            #[cfg(feature = "diagnostics")]
            { *wait = trace.finish(result.as_ref().err().copied()); }
            result
        };
        match taken {
            Ok(taken) => taken,
            Err(why) => {
                DONE[3].fetch_add(1, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                refused(why);
                let _ = why;
                None
            }
        }
    }

    fn take_steps(
        &self,
        written: &[(usize, usize)],
        runs: &[&HostList],
        preplan: Option<Preplan>,
        by_order: bool,
        redirect_fills: bool,
        #[cfg(feature = "diagnostics")] trace: &mut redirect_stats::Attempt,
    ) -> Result<Option<Taken>, &'static str> {
        let mut spent = crate::timeline::span("steps on the card: chosen");
        let stage = stage();
        let (does, steps, chosen, behind) = match preplan {
            Some(p) => (p.does, p.steps, p.chosen, p.behind),
            None => {
                let (does, steps) = does_of(written, runs);
                let mut behind = Behind::default();
                let chosen = plan_with_flights(&does, written, stage, by_order, redirect_fills, &mut behind);
                (does, steps, chosen, behind)
            }
        };
        let chosen = match chosen {
            Ok(chosen) => chosen,
            Err(why) => match retry_cpu_fills(&does, written, stage, by_order, redirect_fills, &behind) {
                Some((chosen, _promoted)) => {
                    #[cfg(feature = "diagnostics")]
                    { trace.promoted = _promoted; }
                    chosen
                }
                None => {
                    #[cfg(feature = "diagnostics")]
                    { trace.cause = behind.cause; }
                    return Err(why);
                }
            },
        };
        if chosen.is_empty() {
            return Ok(None);
        }
        let mapping = chosen.iter().any(|step| step.redirect.is_some())
            .then(crate::watch::changes);
        #[cfg(feature = "diagnostics")]
        if mapping.is_some() {
            trace.planned(&chosen);
        }
        spent.next("steps on the card: what holds their bytes");
        let mut writers = {
            let line = line();
            let mut all = writers(&line.line);
            all.extend(line.landing.iter().map(|w| Writer {
                shadow: w.shadow.clone(),
                ..*w
            }));
            all
        };
        let mut ops: Vec<Op> = Vec::with_capacity(chosen.len());
        #[cfg(feature = "diagnostics")]
        let mut redirected_ops = 0u64;
        // The places read (host copy, address, bytes, level): where in the
        // buffer between their bytes are.
        let mut places: HashMap<(usize, usize, usize, u32), u64> = HashMap::default();
        // Some 2,800 fills a frame, of a few hundred places: what holds each
        // is looked for once, as long as no copy is added to the writers.
        let mut held: HashMap<(usize, usize), Arc<Shadow>> = HashMap::default();
        let mut room = HashMap::default();
        let mut end = 0u64;
        for &Planned { index: i, level, redirect } in &chosen {
            let Some(((src, _), (dst, _))) = reads_writes(&does[i]) else {
                continue;
            };
            let (Does::Copy { len, .. } | Does::Fill { len, .. }) = does[i] else {
                continue;
            };
            if len == 0 {
                return Err("nothing to read");
            }
            let read_src = redirect.map_or(src, |redirect| redirect.address);
            let from = match held.get(&(read_src, len)) {
                Some(from) => from.clone(),
                None => {
                    let from = holder(&writers, read_src, read_src + len)?;
                    held.insert((read_src, len), from.clone());
                    from
                }
            };
            if let Some(redirect) = redirect {
                verify_redirect(runs, steps[redirect.placed], &from)?;
                #[cfg(feature = "diagnostics")]
                { trace.verified += 1; }
            }
            let place = (Arc::as_ptr(&from) as usize, read_src, len, level);
            let at = match places.get(&place) {
                Some(&at) => at,
                None => {
                    if end + len as u64 > BETWEEN {
                        return Err("too much at once");
                    }
                    if !mapped(read_src, read_src + len, &mut room) {
                        return Err("not all of what it reads is mapped");
                    }
                    let at = end;
                    end = (end + len as u64).next_multiple_of(256);
                    places.insert(place, at);
                    at
                }
            };
            let to = match does[i] {
                Does::Copy { .. } => {
                    let to = holder(&writers, dst, dst + len)?;
                    if elsewhere(&does, Arc::as_ptr(&to) as usize, dst, dst + len) {
                        return Err("another host copy of what it writes is read");
                    }
                    if !mapped(dst, dst + len, &mut room) {
                        return Err("not all of what it writes is mapped");
                    }
                    // What the steps behind it read of it is in that host
                    // copy, or nowhere.
                    writers.push(Writer {
                        shadow: Some(to.clone()),
                        from: dst,
                        to: dst + len,
                        copy: true,
                    });
                    held.clear();
                    To::Copy(to, dst)
                }
                _ => match fill_to(runs, steps[i]) {
                    Some(FillTo::Slot(heap, index)) => To::Slot(heap, index),
                    Some(FillTo::Buffer(buffer)) => To::Buffer(buffer),
                    None => {
                        if redirect.is_some() {
                            #[cfg(feature = "diagnostics")]
                            {
                                trace.missing_target += 1;
                            }
                            return Err("a redirected fill has no recorded GPU destination");
                        }
                        continue;
                    }
                },
            };
            #[cfg(feature = "diagnostics")]
            if redirect.is_some() {
                redirected_ops += 1;
            }
            ops.push(Op {
                from: (from, read_src),
                len,
                at,
                level,
                to,
            });
        }
        if mapping.is_some_and(|before| before != crate::watch::changes()) {
            return Err("title memory changed while checking redirected fills");
        }
        spent.next("steps on the card: the list");
        let Some(own) = self.own_list() else {
            return Err("no host list or buffer for it");
        };
        // First every refresh, then the writes: the bytes compared with are
        // the title's. (One step for each host copy, over all that is read of
        // it: a flush brings a copy up to date from the first range asked for
        // to the last, whatever the steps.)
        let mut spans: ByCopy<(Arc<Shadow>, usize, usize)> = ByCopy::default();
        for op in &ops {
            let to = match &op.to {
                To::Copy(to, dst) => Some((to, *dst)),
                _ => None,
            };
            for (shadow, at) in std::iter::once((&op.from.0, op.from.1)).chain(to) {
                let span = spans.entry(Arc::as_ptr(shadow) as usize).or_insert_with(|| (shadow.clone(), at, at + op.len));
                (span.1, span.2) = (span.1.min(at), span.2.max(at + op.len));
            }
        }
        for (shadow, from, to) in spans.into_values() {
            shadow.refresh(&own.list, from as u64, (to - from) as u64);
        }
        let mut copied = Vec::new();
        for op in &ops {
            if let To::Copy(to, dst) = &op.to {
                to.gpu_writes(&own.list, *dst as u64, op.len as u64);
                copied.push((to.clone(), *dst, *dst + op.len));
            }
        }
        #[cfg(feature = "diagnostics")]
        own.list.note_op(|| format!("{} copies of title memory done on the card, {end} bytes (VICTORY_CARD_STEPS)", ops.len()));
        let levels = ops.iter().map(|op| op.level).max().unwrap_or(0);
        for level in 0..=levels {
            let mut reads = Vec::new();
            let mut writes = Vec::new();
            for op in ops.iter().filter(|op| op.level == level) {
                let len = op.len as u64;
                let read = (&op.from.0.buffer, op.from.0.offset(op.from.1 as u64), op.at, len);
                // (Each place once.)
                if places.remove(&(Arc::as_ptr(&op.from.0) as usize, op.from.1, op.len, level)).is_some() {
                    reads.push(read);
                }
                match &op.to {
                    To::Copy(to, dst) => writes.push((&to.buffer, to.offset(*dst as u64), op.at, len)),
                    To::Buffer(buffer) => writes.push((buffer, 0, op.at, len)),
                    To::Slot(..) => {}
                }
            }
            record(level == 0, &own.list.list, &own.between, &reads, &writes);
        }
        own.list.close();
        // SAFETY: a live buffer.
        let gpu = unsafe { own.between.GetGPUVirtualAddress() };
        spent.next("steps on the card: the slots");
        // (11,000 slots a frame in a street, two descriptors each: on
        // several threads, a slot always on the same one, in the steps'
        // order.)
        let parts = if ops.len() < 256 { 1 } else { 8 };
        crate::land::each_under(parts, &|part| {
            for op in &ops {
                if let To::Slot(heap, index) = &op.to
                    && *index as usize / 4 % parts == part
                {
                    // (A heap that is gone: nothing can read the slot.)
                    crate::descriptor::constant_slot(*heap, *index, &own.between, gpu + op.at, op.len as u32);
                }
            }
        });
        let copies = ops.iter().filter(|op| matches!(op.to, To::Copy(..))).count();
        DONE[0].fetch_add((ops.len() - copies) as u64, Ordering::Relaxed);
        DONE[1].fetch_add(copies as u64, Ordering::Relaxed);
        DONE[2].fetch_add(ops.iter().map(|op| op.len as u64).sum(), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            if redirected_ops > 0 {
                trace.executed = redirected_ops;
                PLACED_STATS[0].fetch_add(redirected_ops, Ordering::Relaxed);
            }
            noted(&ops);
        }
        spent.end();
        Ok(Some(Taken {
            own,
            steps: chosen.iter().map(|&Planned { index, .. }| steps[index]).collect(),
            copied,
        }))
    }

    /// A list of ours open for recording, and its buffer.
    fn own_list(&self) -> Option<Own> {
        let pooled = POOL.lock().unwrap_or_else(|e| e.into_inner()).pop();
        if let Some(own) = pooled {
            own.list.reset();
            return Some(own);
        }
        let between = self.uav_buffer(BETWEEN)?;
        name(&between, || "copies of title memory on the card".into());
        let list = self.new_list().ok()?;
        Some(Own { list, between })
    }
}

/// Diagnostics: the copies and fills done on the card since the last
/// frame-rate line, by bytes, host copy read and host copy written (start
/// and length; zeros for a fill): how many, and the last title addresses.
#[cfg(feature = "diagnostics")]
type Noted = std::collections::HashMap<(usize, (u64, u64), (u64, u64)), (u64, usize, usize)>;
#[cfg(feature = "diagnostics")]
static NOTED: Mutex<Option<Noted>> = Mutex::new(None);

#[cfg(feature = "diagnostics")]
fn noted(ops: &[Op]) {
    let mut noted = NOTED.lock().unwrap_or_else(|e| e.into_inner());
    let noted = noted.get_or_insert_with(Default::default);
    for op in ops {
        let (to, dst) = match &op.to {
            To::Copy(to, dst) => (to.span(), *dst),
            To::Slot(..) | To::Buffer(_) => ((0, 0), 0),
        };
        let e = noted.entry((op.len, op.from.0.span(), to)).or_default();
        *e = (e.0 + 1, op.from.1, dst);
    }
}

/// For the frame-rate line.
pub(super) fn log(frames: f64) {
    if stage() == 0 {
        return;
    }
    let d: Vec<f64> = DONE.iter().map(|d| d.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "steps done on the card instead of waiting for flights, per frame (VICTORY_CARD_STEPS={}): {:.2} fills, {:.2} buffer \
         copies, {:.1} KB; {:.2} submissions refused and waited",
        stage(),
        d[0],
        d[1],
        d[2] / 1024.0,
        d[3],
    );
    #[cfg(feature = "diagnostics")]
    {
        redirect_stats::line();
        let redirected = PLACED_STATS[0].swap(0, Ordering::Relaxed) as f64 / frames;
        let cuts = PLACED_STATS[1].swap(0, Ordering::Relaxed) as f64 / frames;
        if redirected > 0.0 || cuts > 0.0 {
            tracing::info!(
                "placed-copy fill read-through per frame (VICTORY_PLACED_ORDER): {:.2} fills redirected, {:.2} cuts",
                redirected,
                cuts,
            );
        }
        let r: Vec<f64> = place::READERS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        if r.iter().any(|&n| n > 0.0) {
            tracing::info!(
                "what a copy at its place writes, found read on the CPU behind it, per frame (looks, all threads): {:.2} by \
                 another such copy, {:.2} by a buffer copy, {:.2} by a fill, {:.2} by a texture, {:.2} by a step of ours",
                r[0],
                r[1],
                r[2],
                r[3],
                r[4]
            );
        }
        let noted = NOTED.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
        let mut noted: Vec<_> = noted.into_iter().collect();
        noted.sort_unstable_by_key(|(_, e)| std::cmp::Reverse(e.0));
        for ((len, from, to), (n, src, dst)) in noted.into_iter().take(5) {
            let n = n as f64 / frames;
            if to.1 == 0 {
                tracing::info!("done on the card, per frame: {n:.2} fills of {len:#x} bytes from the host copy of {:#x}+{:#x} (last {src:#x})", from.0, from.1);
            } else {
                tracing::info!(
                    "done on the card, per frame: {n:.2} copies of {len:#x} bytes from the host copy of {:#x}+{:#x} (last {src:#x}) \
                     to that of {:#x}+{:#x} (last {dst:#x})",
                    from.0,
                    from.1,
                    to.0,
                    to.1
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(
        does: &[Does],
        stage: u32,
        by_order: bool,
        behind: &mut Option<(usize, usize)>,
    ) -> Result<Vec<Planned>, &'static str> {
        let mut observation = Behind {
            steps: *behind,
            ..Behind::default()
        };
        let result = super::plan(does, stage, by_order, true, &mut observation);
        *behind = observation.steps;
        result
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn redirect_refusals_explain_the_same_blocking_steps() {
        let copy_on_card = Does::Placed { dst: 200, src: 100, len: 16, from_cpu: false };
        let upload = Does::Placed { dst: 200, src: 100, len: 16, from_cpu: true };
        let partial = Does::Placed { dst: 200, src: 500, len: 4, from_cpu: false };
        type Case<'a> = (&'a [Does], &'a [(usize, usize)], (usize, usize), &'static str);
        let cases: &[Case<'_>] = &[
            (&[upload, fill(200, 8, true)], &[], (0, 1), "upload_source"),
            (&[copy_on_card, fill(212, 8, true)], &[], (0, 1), "partial_coverage"),
            (&[copy_on_card, partial, fill(200, 8, true)], &[], (0, 2), "partial_newer_copy"),
            (&[upload, fill(400, 4, true)], &[(100, 116)], (0, 0), "pending_upload_source"),
            (&[copy_on_card, fill(200, 4, true), fill(204, 4, false)], &[(100, 116)], (0, 2), ELIGIBLE_CPU_FILL),
            (&[copy_on_card, fill(200, 4, true), stuck(204, 4)], &[(100, 116)], (0, 2), "remaining_cpu_fill_cpu_only"),
            (&[copy_on_card, fill(200, 4, true), fill(212, 8, false)], &[(100, 116)], (0, 2), "remaining_cpu_fill_without_covering_copy"),
        ];
        for &(steps, air, pair, cause) in cases {
            let mut behind = Behind::default();
            assert_eq!(plan_with_flights(steps, air, 2, true, true, &mut behind), Err(BEHIND_PLACED));
            assert_eq!(behind.steps, Some(pair));
            assert_eq!(behind.cause, Some(cause));
        }
    }

    #[test]
    fn retry_redirects_a_second_fill_whose_range_is_not_in_the_flights() {
        let placed = Does::Placed { dst: 200, src: 100, len: 16, from_cpu: false };
        let does = [placed, fill(200, 4, true), fill(204, 4, false)];
        let air = [(100, 116), (200, 204)];
        assert!(!touches(&air, 204, 208));
        let mut refusal = Behind::default();
        assert_eq!(plan_with_flights(&does, &air, 2, true, true, &mut refusal), Err(BEHIND_PLACED));
        assert_eq!(refusal.steps, Some((0, 2)));
        let (chosen, promoted) = retry_cpu_fills(&does, &air, 2, true, true, &refusal).unwrap();
        assert_eq!(promoted, 1);
        assert_eq!(chosen, [
            Planned { index: 1, level: 0, redirect: Some(Redirect { address: 100, placed: 0 }) },
            Planned { index: 2, level: 0, redirect: Some(Redirect { address: 104, placed: 0 }) },
        ]);
        assert!(retry_cpu_fills(&does, &air, 2, true, false, &refusal).is_none());
    }

    #[test]
    fn retry_keeps_the_original_refusal_if_a_cpu_only_fill_remains() {
        let placed = Does::Placed { dst: 200, src: 100, len: 16, from_cpu: false };
        let does = [placed, fill(200, 4, true), fill(204, 4, false), stuck(208, 4)];
        let air = [(100, 116), (200, 204)];
        let mut refusal = Behind::default();
        assert_eq!(plan_with_flights(&does, &air, 2, true, true, &mut refusal), Err(BEHIND_PLACED));
        assert_eq!(refusal.cause, Some(ELIGIBLE_CPU_FILL));
        let promoted = retry_cpu_fills(&does, &air, 2, true, true, &refusal);
        assert!(promoted.is_none());
        assert_eq!(refusal.steps, Some((0, 2)));
    }

    #[test]
    fn retry_counts_only_fills_left_out_of_the_first_plan() {
        let placed = Does::Placed { dst: 200, src: 100, len: 16, from_cpu: false };
        let does = [copy(200, 400, 4, true), placed, fill(200, 4, false), fill(204, 4, false)];
        let air = [(100, 116), (400, 404)];
        let mut refusal = Behind::default();
        assert_eq!(plan_with_flights(&does, &air, 2, true, true, &mut refusal), Err(BEHIND_PLACED));
        assert_eq!(refusal.steps, Some((1, 3)));
        let (chosen, promoted) = retry_cpu_fills(&does, &air, 2, true, true, &refusal).unwrap();
        assert_eq!(promoted, 1);
        assert_eq!(chosen.iter().map(|p| p.index).collect::<Vec<_>>(), [0, 2, 3]);
    }

    #[test]
    fn retry_never_skips_the_earlier_write_to_the_recorded_source() {
        let placed = Does::Placed { dst: 200, src: 100, len: 16, from_cpu: false };
        let does = [
            Does::GpuWrite { from: 104, to: 108 },
            fill(400, 4, true),
            placed,
            fill(204, 4, false),
        ];
        let air = [(100, 116), (400, 404)];
        let mut refusal = Behind::default();
        assert_eq!(plan_with_flights(&does, &air, 2, true, true, &mut refusal), Err(BEHIND_PLACED));
        assert_eq!(refusal.steps, Some((2, 3)));
        assert_eq!(refusal.cause, Some(ELIGIBLE_CPU_FILL));
        assert!(retry_cpu_fills(&does, &air, 2, true, true, &refusal).is_none());
    }

    #[test]
    fn read_through_still_waits_for_an_untaken_cpu_reader_of_a_pending_copy() {
        let placed = Does::Placed { dst: 200, src: 100, len: 16, from_cpu: false };
        let redirected = fill(200, 4, true);
        let air = [(100, 116), (200, 204)];
        assert_eq!(
            plan_with_flights(&[placed, redirected], &air, 2, true, true, &mut Behind::default()),
            Ok(vec![Planned { index: 1, level: 0, redirect: Some(Redirect { address: 100, placed: 0 }) }]),
        );
        for reader in [
            Does::Read { from: 204, to: 208, asked: false },
            fill(204, 4, false),
            copy(400, 204, 4, false),
            Does::Placed { dst: 400, src: 204, len: 4, from_cpu: true },
        ] {
            let mut behind = Behind::default();
            assert_eq!(
                plan_with_flights(&[placed, redirected, reader], &air, 2, true, true, &mut behind),
                Err(BEHIND_PLACED),
                "{reader:?}",
            );
            assert_eq!(behind.steps, Some((0, 2)));
        }
    }

    #[test]
    fn read_through_requires_the_recorded_backing_even_if_the_ranges_match() {
        let recorded = Arc::new((100, 116));
        let another = Arc::new((100, 116));
        let same = recorded.clone();
        assert_eq!(require_recorded_source(Some(&recorded), &same), Ok(()));
        assert!(require_recorded_source(Some(&recorded), &another).is_err());
        assert!(require_recorded_source(None, &same).is_err());
    }

    #[test]
    fn disabling_read_through_keeps_the_original_placed_copy_wait() {
        let placed = Does::Placed { dst: 200, src: 100, len: 16, from_cpu: false };
        let mut behind = Behind::default();
        assert_eq!(
            super::plan(&[placed, fill(200, 4, true)], 2, true, false, &mut behind),
            Err(BEHIND_PLACED),
        );
        assert_eq!(behind.steps, Some((0, 1)));
    }

    fn copy(dst: usize, src: usize, len: usize, asked: bool) -> Does {
        Does::Copy { dst, src, len, asked }
    }

    fn fill(src: usize, len: usize, asked: bool) -> Does {
        Does::Fill { src, len, asked, to: true }
    }

    fn stuck(src: usize, len: usize) -> Does {
        Does::Fill { src, len, asked: false, to: false }
    }

    /// What is done on the card, as (place, level), in stage 2.
    fn planned(does: &[Does]) -> Vec<(usize, u32)> {
        plan(does, 2, true, &mut None).unwrap().into_iter().map(|p| (p.index, p.level)).collect()
    }

    /// Only what is asked for is taken, all of it, and a step that stands
    /// in none of their ways stays where it is.
    #[test]
    fn steps_apart_from_one_another_are_taken() {
        let does = [
            copy(1000, 5000, 100, false),
            fill(2000, 64, true),
            Does::Refresh { shadow: 1, from: 0, to: 9000 },
            copy(3000, 6000, 100, true),
            fill(1000, 64, false),
            Does::Read { from: 3100, to: 4000, asked: false },
            stuck(3100, 5),
        ];
        assert_eq!(planned(&does), [(1, 0), (3, 0)]);
        // Two fills of the same bytes.
        assert_eq!(
            plan(&[fill(0, 64, true), fill(32, 64, true)], 1, true, &mut None),
            Ok(vec![
                Planned { index: 0, level: 0, redirect: None },
                Planned { index: 1, level: 0, redirect: None },
            ])
        );
        assert_eq!(plan(&[fill(0, 64, false)], 1, true, &mut None), Ok(vec![]));
        // Ends that touch are apart.
        let does = [
            copy(1000, 5000, 100, true),
            copy(900, 9000, 100, false),
            copy(1100, 9000, 5, false),
            copy(4900, 9000, 100, false),
        ];
        assert_eq!(planned(&does), [(0, 0)]);
    }

    /// What depends on a taken step, or it on it, is taken too, on the level
    /// the order of the steps asks for.
    #[test]
    fn steps_that_depend_on_one_another_are_taken_in_order() {
        let taken = copy(1000, 5000, 100, true);
        // Reads what it writes, behind and ahead of it.
        assert_eq!(planned(&[taken, fill(1099, 64, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[fill(1099, 64, false), taken]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(7000, 1050, 10, false)]), [(0, 0), (1, 1)]);
        // Writes what it reads, and what it writes.
        assert_eq!(planned(&[copy(5099, 9000, 1, false), taken]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(5099, 9000, 1, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(999, 9000, 2, false)]), [(0, 0), (1, 1)]);
        // The same list twice.
        assert_eq!(planned(&[taken, taken]), [(0, 0), (1, 1)]);
        // A chain, with steps apart from it in between, and two fills of one
        // copy on one level.
        let does = [
            copy(8000, 7000, 50, false),
            taken,
            copy(2000, 1010, 20, false),
            fill(90_000, 10, false),
            fill(2000, 20, false),
            fill(2010, 20, true),
            copy(1000, 2005, 5, false),
            fill(1000, 100, false),
        ];
        assert_eq!(planned(&does), [(1, 0), (2, 1), (4, 2), (5, 2), (6, 2), (7, 3)]);
        // A fill is in nobody's way.
        assert_eq!(planned(&[taken, fill(5000, 100, false)]), [(0, 0)]);
        // A copy ahead that a taken one only shares its source with.
        assert_eq!(planned(&[copy(7000, 5000, 10, false), taken]), [(1, 0)]);
    }

    /// A copy at its place in a list is in the way of what reads or writes its
    /// bytes behind it, not of what did ahead of it — except a taken fill
    /// wholly inside a card-sourced placed copy's destination, which redirects
    /// to the placed copy's source host copy when unchained.
    #[test]
    fn copies_at_their_place_stand_in_the_way_of_what_is_behind_them() {
        let placed = Does::Placed { dst: 2000, src: 3000, len: 64, from_cpu: false };
        let placed_cpu = Does::Placed { dst: 2000, src: 3000, len: 64, from_cpu: true };
        let behind = |does: &[Does]| {
            let mut found = None;
            let refused = plan(does, 2, true, &mut found);
            assert!(refused.is_ok() || found.is_some() && refused == Err(BEHIND_PLACED));
            assert_eq!(plan(does, 2, false, &mut None).is_err(), does.len() > 1 && !matches!(does, [_, Does::Fill { src: 5000, .. }]));
            found
        };
        assert_eq!(behind(&[fill(2032, 64, true), placed]), None);
        // Partial overlap with placed copy's destination: still refused.
        assert_eq!(behind(&[placed, fill(2032, 64, true)]), Some((0, 1)));
        assert_eq!(behind(&[placed, fill(5000, 64, true)]), None);
        // Wholly inside a card-sourced placed copy's destination: redirected to `src + offset`.
        assert_eq!(
            plan(&[fill(2000, 8, true), placed, fill(2016, 16, true)], 2, true, &mut None),
            Ok(vec![
                Planned { index: 0, level: 0, redirect: None },
                Planned { index: 2, level: 0, redirect: Some(Redirect { address: 3016, placed: 1 }) },
            ])
        );
        // Without `by_order`, even a wholly-contained fill is still refused.
        assert_eq!(plan(&[placed, fill(2016, 16, true)], 2, false, &mut None), Err(BEHIND_PLACED));
        // CPU-sourced placed copy: refused (no source host copy on the card).
        assert_eq!(behind(&[placed_cpu, fill(2000, 8, true)]), Some((0, 1)));
        // Chained placed copy (earlier placed or copy writes `placed.src`): refused.
        let chain_placed = Does::Placed { dst: 3000, src: 4000, len: 64, from_cpu: false };
        assert_eq!(behind(&[chain_placed, placed, fill(2000, 8, true)]), Some((1, 2)));
        assert_eq!(behind(&[copy(3000, 5000, 8, false), placed, fill(2000, 8, true)]), Some((1, 2)));
        assert_eq!(behind(&[placed, fill(2000, 8, true), copy(3000, 5000, 8, false)]), Some((0, 1)));
        // Two placed copies to the same dst: fill wholly inside the newer one redirects to the newer source.
        let placed2 = Does::Placed { dst: 2000, src: 4000, len: 64, from_cpu: false };
        assert_eq!(
            plan(&[placed, placed2, fill(2008, 16, true)], 2, true, &mut None),
            Ok(vec![Planned { index: 2, level: 0, redirect: Some(Redirect { address: 4008, placed: 1 }) }])
        );
        // Newer placed copy only partially covers the fill (split across two placed copies): refused.
        let placed_half = Does::Placed { dst: 2000, src: 4000, len: 32, from_cpu: false };
        assert_eq!(behind(&[placed, placed_half, fill(2000, 64, true)]), Some((0, 2)));
        // Shader write ahead of `placed` touching `placed.src`: refused (cannot read pre-dispatch bytes in front list).
        assert_eq!(behind(&[Does::GpuWrite { from: 3000, to: 3016 }, placed, fill(2000, 16, true)]), Some((1, 2)));
        // Shader write *after* `placed` touching `placed.src`: safe to redirect (front list runs before it).
        assert_eq!(
            plan(&[placed, Does::GpuWrite { from: 3000, to: 3016 }, fill(2000, 16, true)], 2, true, &mut None),
            Ok(vec![Planned { index: 2, level: 0, redirect: Some(Redirect { address: 3000, placed: 0 }) }])
        );
        // Shader write between `placed` and `fill` touching `fill`'s range: refused.
        assert_eq!(behind(&[placed, Does::GpuWrite { from: 2008, to: 2024 }, fill(2000, 16, true)]), Some((0, 2)));
        // One that writes what it copies from, and one that writes over it.
        assert_eq!(behind(&[copy(3000, 5000, 8, true), placed]), None);
        assert_eq!(behind(&[placed, copy(3000, 5000, 8, true)]), Some((0, 1)));
        assert_eq!(behind(&[copy(2060, 5000, 8, true), placed]), None);
        assert_eq!(behind(&[placed, copy(2060, 5000, 8, true)]), Some((0, 1)));
    }

    /// Regression cases for placed-copy source substitution, kept separate
    /// from placement-order performance checks: partial overlaps on every
    /// boundary and later writes that cover the full fill range.
    #[test]
    fn placed_redirect_regression_cases() {
        let placed = Does::Placed { dst: 2000, src: 3000, len: 64, from_cpu: false };
        let fill_mid = fill(2016, 16, true);
        let expect_refused = |does: &[Does], label: &str| {
            let mut behind = None;
            assert_eq!(plan(does, 2, true, &mut behind), Err(BEHIND_PLACED), "{label}");
            assert!(behind.is_some(), "{label}: missing behind pair");
        };

        // Partial overlaps: leading edge, trailing edge, and fill wider than the placed copy.
        expect_refused(&[placed, fill(1992, 16, true)], "leading partial overlap");
        expect_refused(&[placed, fill(2056, 16, true)], "trailing partial overlap");
        expect_refused(&[placed, fill(1992, 80, true)], "fill wider than placed copy");

        // Later buffer copy covering the full fill (exact and wider): refused.
        expect_refused(&[placed, copy(2016, 5000, 16, false), fill_mid], "later exact buffer copy covers fill");
        expect_refused(&[placed, copy(2000, 5000, 64, false), fill_mid], "later wide buffer copy covers fill");

        // Later shader write covering the full fill (exact and wider): refused.
        expect_refused(&[placed, Does::GpuWrite { from: 2016, to: 2032 }, fill_mid], "later exact gpu write covers fill");
        expect_refused(&[placed, Does::GpuWrite { from: 2000, to: 2064 }, fill_mid], "later wide gpu write covers fill");

        // Later card-sourced placed copy covering the full fill: redirects to the newer source, not the older one.
        let newer_exact = Does::Placed { dst: 2016, src: 4000, len: 16, from_cpu: false };
        assert_eq!(
            plan(&[placed, newer_exact, fill_mid], 2, true, &mut None),
            Ok(vec![Planned { index: 2, level: 0, redirect: Some(Redirect { address: 4000, placed: 1 }) }])
        );
        let newer_wide = Does::Placed { dst: 2000, src: 4000, len: 64, from_cpu: false };
        assert_eq!(
            plan(&[placed, newer_wide, fill_mid], 2, true, &mut None),
            Ok(vec![Planned { index: 2, level: 0, redirect: Some(Redirect { address: 4016, placed: 1 }) }])
        );

        // Later placed copy covering the full fill whose source is invalid (CPU-sourced or written earlier): refused.
        let newer_cpu = Does::Placed { dst: 2000, src: 4000, len: 64, from_cpu: true };
        expect_refused(&[placed, newer_cpu, fill_mid], "later cpu-sourced placed copy covers fill");
        expect_refused(
            &[Does::GpuWrite { from: 4000, to: 4064 }, placed, newer_wide, fill_mid],
            "later placed copy covers fill but its source was written earlier",
        );
        let chain_for_newer = Does::Placed { dst: 4000, src: 6000, len: 64, from_cpu: false };
        expect_refused(
            &[chain_for_newer, placed, newer_wide, fill_mid],
            "later placed copy covers fill but is chained to an earlier placed copy",
        );

        // Earlier write invalidating `placed`'s source slice for the fill range: refused.
        expect_refused(
            &[Does::GpuWrite { from: 3016, to: 3032 }, placed, fill_mid],
            "earlier gpu write invalidates redirected source",
        );
        expect_refused(
            &[copy(3016, 5000, 16, false), placed, fill_mid],
            "earlier buffer copy invalidates redirected source",
        );
        // Earlier write to `placed`'s source outside the redirected slice: conservatively refused.
        expect_refused(
            &[Does::GpuWrite { from: 3000, to: 3008 }, placed, fill_mid],
            "earlier gpu write to placed source outside fill slice",
        );

        // Split cover: a partial later `Placed` plus a `GpuWrite` covering the rest (either order): refused.
        let newer_left = Does::Placed { dst: 2016, src: 4000, len: 8, from_cpu: false };
        let gpu_right = Does::GpuWrite { from: 2024, to: 2032 };
        expect_refused(&[placed, newer_left, gpu_right, fill_mid], "split cover: partial placed then gpu write");
        expect_refused(&[placed, gpu_right, newer_left, fill_mid], "split cover: gpu write then partial placed");

        // Later `Placed` (after `placed`) writing `placed`'s source: allowed because the front list reads first.
        let later_writes_src = Does::Placed { dst: 3000, src: 5000, len: 64, from_cpu: false };
        assert_eq!(
            plan(&[placed, later_writes_src, fill_mid], 2, true, &mut None),
            Ok(vec![Planned { index: 2, level: 0, redirect: Some(Redirect { address: 3016, placed: 0 }) }])
        );

        // Self-overlapping `Placed`: standalone redirect reads pre-copy source;
        // with the `GpuWrite(dst)` that `place_copy` declares ahead of it, refused.
        let self_overlap = Does::Placed { dst: 2000, src: 2016, len: 64, from_cpu: false };
        assert_eq!(
            plan(&[self_overlap, fill(2000, 16, true)], 2, true, &mut None),
            Ok(vec![Planned { index: 1, level: 0, redirect: Some(Redirect { address: 2016, placed: 0 }) }])
        );
        expect_refused(
            &[Does::GpuWrite { from: 2000, to: 2064 }, self_overlap, fill(2000, 16, true)],
            "self-overlapping placed copy preceded by declared dst gpu write",
        );
    }

    /// Randomized equivalence check with `Does::Placed` and `Does::GpuWrite`:
    /// redirected fills read the exact bytes sequential execution produces.
    #[test]
    fn placed_redirects_match_sequential_order() {
        let mut seed = 98765u32;
        let mut random = move |n: u32| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) % n
        };
        let (mut redirected_cases, mut refused) = (0, 0);
        for case in 0..8000u32 {
            let does: Vec<Does> = (0..2 + random(7))
                .map(|_| {
                    let (a, b, len) = (random(40) as usize, random(40) as usize, 1 + random(8) as usize);
                    match random(7) {
                        0..2 => Does::Placed { dst: a, src: b, len, from_cpu: random(5) == 0 },
                        2..4 => fill(a, len, random(2) == 0),
                        4 => copy(a, b, len, random(3) == 0),
                        5 => Does::GpuWrite { from: a, to: a + len },
                        _ => Does::Read { from: a, to: a + len, asked: false },
                    }
                })
                .collect();
            let Ok(chosen) = plan(&does, 2, true, &mut None) else {
                refused += 1;
                continue;
            };
            if chosen.iter().any(|p| p.redirect.is_some()) {
                redirected_cases += 1;
            }
            let start: Vec<u32> = (0..64).collect();
            // Sequential execution: what each step reads (with shaders writing unique words).
            let mut seq = start.clone();
            let mut wanted: Vec<Option<Vec<u32>>> = Vec::new();
            for (step_idx, d) in does.iter().enumerate() {
                match *d {
                    Does::Copy { dst, src, len, .. } => {
                        wanted.push(Some(seq[src..src + len].to_vec()));
                        seq.copy_within(src..src + len, dst);
                    }
                    Does::Placed { dst, src, len, .. } => {
                        wanted.push(None);
                        seq.copy_within(src..src + len, dst);
                    }
                    Does::GpuWrite { from, to } => {
                        wanted.push(None);
                        for (off, slot) in seq[from..to].iter_mut().enumerate() {
                            *slot = 0x1000_0000 ^ (case << 12) ^ ((step_idx as u32) << 6) ^ off as u32;
                        }
                    }
                    Does::Fill { src, len, .. } => wanted.push(Some(seq[src..src + len].to_vec())),
                    Does::Read { from, to, .. } => wanted.push(Some(seq[from..to].to_vec())),
                    _ => wanted.push(None),
                }
            }
            // Host execution: untaken CPU copies first, then front list (card steps
            // level by level with redirects), then placed copies and shaders at their places.
            let mut mem = start.clone();
            let is_taken = |i: usize| chosen.iter().any(|p| p.index == i);
            for (i, d) in does.iter().enumerate() {
                if !is_taken(i)
                    && let Does::Copy { dst, src, len, .. } = *d
                {
                    mem.copy_within(src..src + len, dst);
                }
            }
            // Verify every redirected fill reads exact sequential bytes (even in
            // the presence of arbitrary `GpuWrite`s), and every taken step does
            // when the sequence has no `GpuWrite`s.
            let has_gpu_write = does.iter().any(|d| matches!(d, Does::GpuWrite { .. }));
            for level in 0..=chosen.iter().map(|p| p.level).max().unwrap_or(0) {
                let before = mem.clone();
                for &Planned { index, redirect, .. } in chosen.iter().filter(|p| p.level == level) {
                    let ((src, end), (dst, _)) = reads_writes(&does[index]).unwrap();
                    let read_src = redirect.map_or(src, |redirect| redirect.address);
                    let len = end - src;
                    if redirect.is_some() || !has_gpu_write {
                        assert_eq!(
                            before[read_src..read_src + len],
                            wanted[index].as_deref().unwrap()[..],
                            "step {index} of {does:?}: {chosen:?}"
                        );
                    }
                    if matches!(does[index], Does::Copy { .. }) {
                        mem[dst..dst + len].copy_from_slice(&before[read_src..read_src + len]);
                    }
                }
            }
        }
        assert!(redirected_cases > 50 && refused > 50, "{redirected_cases} {refused}");
    }

    /// Whatever the steps, done level by level (in a level every read
    /// before every write) they read and leave what they read and leave one
    /// after the other, the CPU's steps ahead of all of them.
    #[test]
    fn levels_do_what_the_order_does() {
        let mut seed = 12345u32;
        let mut random = move |n: u32| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) % n
        };
        let (mut done, mut refused) = (0, 0);
        for _ in 0..4000 {
            let does: Vec<Does> = (0..2 + random(9))
                .map(|_| {
                    let (at, other, len) = (random(60) as usize, random(60) as usize, 1 + random(8) as usize);
                    let asked = random(4) == 0;
                    match random(8) {
                        0..4 => copy(at, other, len, asked),
                        4..6 => fill(at, len, asked),
                        6 => stuck(at, len),
                        _ => Does::Read { from: at, to: at + len, asked: false },
                    }
                })
                .collect();
            let Ok(chosen) = plan(&does, 2, true, &mut None) else {
                refused += 1;
                continue;
            };
            done += 1;
            let start: Vec<u32> = (0..80).collect();
            // One after the other: what each reads, and what is left.
            let mut memory = start.clone();
            let mut wanted = Vec::new();
            for d in &does {
                let (from, to) = match *d {
                    Does::Read { from, to, .. } => (from, to),
                    _ => reads_writes(d).unwrap().0,
                };
                wanted.push(memory[from..to].to_vec());
                if let Does::Copy { dst, src, len, .. } = *d {
                    memory.copy_within(src..src + len, dst);
                }
            }
            // The CPU's, then the card's.
            let mut cpu = start.clone();
            let taken = |i: usize| chosen.iter().any(|c| c.index == i);
            for (i, d) in does.iter().enumerate().filter(|&(i, _)| !taken(i)) {
                let (from, to) = match *d {
                    Does::Read { from, to, .. } => (from, to),
                    _ => reads_writes(d).unwrap().0,
                };
                assert_eq!(cpu[from..to], wanted[i][..], "step {i} of {does:?}");
                if let Does::Copy { dst, src, len, .. } = *d {
                    cpu.copy_within(src..src + len, dst);
                }
            }
            let mut card = cpu.clone();
            for level in 0..=chosen.iter().map(|c| c.level).max().unwrap_or(0) {
                let of: Vec<usize> = chosen.iter().filter(|c| c.level == level).map(|c| c.index).collect();
                let before = card.clone();
                for i in of {
                    let ((from, to), (dst, _)) = reads_writes(&does[i]).unwrap();
                    assert_eq!(before[from..to], wanted[i][..], "step {i} of {does:?}: {chosen:?}");
                    if matches!(does[i], Does::Copy { .. }) {
                        card[dst..dst + to - from].copy_from_slice(&before[from..to]);
                    }
                }
            }
            assert_eq!(card, memory, "{does:?}: {chosen:?}");
        }
        assert!(done > 500 && refused > 100, "{done} {refused}");
    }

    /// What cannot be done on the card refuses the lot, where it stands in
    /// the way.
    #[test]
    fn steps_in_the_way_are_refused() {
        let taken = copy(1000, 5000, 100, true);
        assert!(plan(&[taken, Does::Run], 2, true, &mut None).is_err());
        // A texture that itself waits for flights; a fill that cannot be
        // taken but waits.
        assert!(plan(&[taken, Does::Read { from: 1, to: 2, asked: true }], 2, true, &mut None).is_err());
        assert!(plan(&[Does::Fill { src: 1, len: 1, asked: true, to: false }], 2, true, &mut None).is_err());
        // Reads on the CPU of what it writes: behind it, not ahead of it.
        for read in [Does::Read { from: 900, to: 1001, asked: false }, stuck(1050, 3)] {
            assert!(plan(&[taken, read], 2, true, &mut None).is_err(), "{read:?} behind");
            assert_eq!(planned(&[read, taken]), [(1, 0)], "{read:?} ahead");
        }
        // Also of what is taken along.
        assert!(plan(&[taken, copy(3000, 1000, 10, false), stuck(3005, 1)], 2, true, &mut None).is_err());
        // Into itself, and copies without stage 2.
        assert!(plan(&[copy(1000, 1050, 100, true)], 2, true, &mut None).is_err());
        assert!(plan(&[taken], 1, true, &mut None).is_err());
        assert!(plan(&[fill(1000, 4, true), copy(1000, 5000, 100, false)], 1, true, &mut None).is_err());
    }

    #[test]
    fn a_refresh_of_another_host_copy_is_seen() {
        let does = [Does::Refresh { shadow: 7, from: 100, to: 200 }];
        assert!(!elsewhere(&does, 7, 100, 200));
        assert!(elsewhere(&does, 8, 199, 300));
        assert!(!elsewhere(&does, 8, 200, 300));
    }

    /// A value written behind a flight, or a read-back, is in no host copy
    /// before it lands.
    #[test]
    fn what_no_host_copy_holds_is_refused() {
        let writers = [Writer { shadow: None, from: 100, to: 108, copy: false }];
        assert!(holder(&writers, 104, 120).is_err());
    }

    /// Reads are all before writes, between two buffers and within one, and
    /// the buffers are usable in the lists that follow in the same call.
    #[test]
    fn the_card_copies_what_is_recorded() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let (a, a_data) = host.shared_buffer(1 << 16).unwrap();
        let (b, b_data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: mappings of 64 KiB.
        let (a_bytes, b_bytes) =
            unsafe { (std::slice::from_raw_parts_mut(a_data, 1 << 16), std::slice::from_raw_parts_mut(b_data, 1 << 16)) };
        for (i, (x, y)) in a_bytes.iter_mut().zip(b_bytes.iter_mut()).enumerate() {
            (*x, *y) = (i as u8 ^ (i >> 8) as u8, 0xee);
        }
        let was = a_bytes.to_vec();
        let own = host.own_list().unwrap();
        // Within `a` (there and back), from `a` to `b`, and a read alone.
        record(
            true,
            &own.list.list,
            &own.between,
            &[(&a, 33, 0, 100), (&a, 2000, 256, 100), (&a, 5001, 512, 7), (&a, 9, 1024, 4)],
            &[(&a, 2000, 0, 100), (&a, 33, 256, 100), (&b, 77, 512, 7)],
        );
        // A level behind it, of what it wrote.
        record(false, &own.list.list, &own.between, &[(&b, 77, 2048, 7)], &[(&b, 1077, 2048, 7)]);
        own.list.close();
        // A list behind it that uses the buffers in other ways.
        let next = host.new_list().unwrap();
        let (out, out_data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: an open list; live buffers.
        unsafe {
            next.list.CopyBufferRegion(&out, 0, &own.between, 1024, 4);
            next.list.CopyBufferRegion(&b, 4096, &a, 33, 100);
        }
        next.close();
        host.execute(&[&own.list, &next]);
        // With the debug layer (`VICTORY_D3D12_DEBUG=1`, diagnostics).
        assert_eq!(host.log_messages(), None);
        assert_eq!(a_bytes[2000..2100], was[33..133]);
        assert_eq!(a_bytes[33..133], was[2000..2100]);
        assert_eq!(b_bytes[77..84], was[5001..5008]);
        assert_eq!(b_bytes[1077..1084], was[5001..5008]);
        assert_eq!(b_bytes[4096..4196], was[2000..2100]);
        // SAFETY: a mapping of 64 KiB.
        assert_eq!(unsafe { std::slice::from_raw_parts(out_data, 4) }, &was[9..13]);
        assert!(b_bytes[..77].iter().all(|&x| x == 0xee));
        a_bytes[2000..2100].copy_from_slice(&was[2000..2100]);
        a_bytes[33..133].copy_from_slice(&was[33..133]);
        assert_eq!(a_bytes, &was[..]);
        // As the landing does: its next use records anew.
        POOL.lock().unwrap().push(own);
        let own = host.own_list().unwrap();
        own.list.close();
    }

    /// The whole way, in a process of its own with the relay on:
    /// `VICTORY_RELAY=2 VICTORY_CARD_STEPS=2 cargo test -p victory-gpu --lib
    /// card_steps -- --ignored --test-threads=1`. A flight's list writes
    /// part of what the next submission's copy reads, the title wrote the
    /// rest; the copy is done on the card behind the flight, and a second
    /// one reads what the first wrote. Title memory ends as on the CPU.
    #[test]
    #[ignore = "wants VICTORY_RELAY=2 VICTORY_CARD_STEPS=2 and a process of its own"]
    fn a_copy_behind_a_flight_reads_what_the_flight_wrote() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const LEN: usize = 4 << 20;
        // Without it (`TEST_UNWATCHED=1`) the title's bytes cannot be put in
        // under a flight's writes: the flights that declared them are waited
        // for.
        let write_watch = if std::env::var_os("TEST_UNWATCHED").is_some() { 0 } else { 0x20_0000 };
        let Some(host) = host() else {
            return;
        };
        assert!(relay_exact() && stage() == 2, "set VICTORY_RELAY=2 VICTORY_CARD_STEPS=2");
        // Two title buffers: in the first a copy within it, from it into the
        // second, and back.
        let (title, start) = {
            // SAFETY: memory of our own, never freed.
            let base = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | write_watch, PAGE_READWRITE) }.cast::<u8>();
            assert!(!base.is_null());
            // SAFETY: `LEN` bytes just committed.
            (unsafe { std::slice::from_raw_parts_mut(base, LEN) }, base as usize)
        };
        const HALF: usize = LEN / 2;
        crate::shadow::register(0x1111, start as u64, HALF as u64);
        crate::shadow::register(0x2222, (start + HALF) as u64, HALF as u64);
        let first = crate::shadow::find(start as u64, 64).unwrap();
        let (slow_from, slow_to) = (host.uav_buffer(64 << 20).unwrap(), host.uav_buffer(64 << 20).unwrap());
        let before = DONE[1].load(Ordering::Relaxed);
        let mut lists = Vec::new();
        for round in 0..8usize {
            // The title writes; then a "shader" of a flight writes 300 bytes
            // of what the copy reads, behind some work.
            let seed = |i: usize| (i * 7 + round * 13 + i / 251) as u8;
            for (i, b) in title[4000..6000].iter_mut().enumerate() {
                *b = seed(i);
            }
            let (upload, data) = host.upload_buffer(4096).unwrap();
            for i in 0..300 {
                // SAFETY: inside the mapping.
                unsafe { data.add(i).write(0x80 | (round * 31 + i) as u8) };
            }
            let a = host.new_list().unwrap();
            first.refresh(&a, start as u64, HALF as u64);
            first.gpu_writes(&a, start as u64, HALF as u64);
            // SAFETY: an open list; live buffers.
            unsafe {
                for _ in 0..24 {
                    a.list.CopyBufferRegion(&slow_to, 0, &slow_from, 0, 64 << 20);
                }
                a.list.CopyBufferRegion(&first.buffer, 4500, &upload, 0, 300);
                a.list
                    .ResourceBarrier(&[transition(&first.buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
            }
            a.close();
            host.execute_own(&a);
            // With the flight in the air, beside what it writes.
            for (i, b) in title[5300..5400].iter_mut().enumerate() {
                *b = seed(1300 + i) ^ 0x5c;
            }
            // Thousands of runs of bytes of its own.
            for b in title[1_700_000..1_708_000].iter_mut().step_by(2) {
                *b = 1 + round as u8;
            }
            // Into the second buffer, and within the first.
            let b = host.new_list().unwrap();
            b.defer_copy((start + HALF + 100) as *mut u8, (start + 4001) as *const u8, 1500);
            b.defer_copy((start + 900_001) as *mut u8, (start + 4400) as *const u8, 777);
            b.defer_copy((start + HALF + 10_000) as *mut u8, (start + 1_700_000) as *const u8, 8000);
            // What depends on them: a copy of the first copy, upload memory
            // filled from that, and the first one's bytes copied over.
            b.defer_copy((start + 1_500_000) as *mut u8, (start + HALF + 600) as *const u8, 700);
            let (staging, staged) = b.upload(host, 4096).unwrap();
            b.defer_fill_buffer(&staging, staged, (start + 1_500_100) as *const u8, 500);
            b.defer_copy((start + HALF + 300) as *mut u8, (start + 8000) as *const u8, 64);
            b.close();
            host.execute_own(&b);
            // What the first wrote, read by the next submission.
            let c = host.new_list().unwrap();
            c.defer_copy((start + 1_200_000) as *mut u8, (start + HALF + 100) as *const u8, 1500);
            c.close();
            let (number, _) = host.execute_own(&c);
            settle_now(WHY_SYNC);
            assert!(FINISHED.load(Ordering::Acquire) >= number);
            let wanted: Vec<u8> = (0..2000)
                .map(|i| match i {
                    500..800 => 0x80 | (round * 31 + i - 500) as u8,
                    1300..1400 => seed(i) ^ 0x5c,
                    _ => seed(i),
                })
                .collect();
            assert!(title[4000..6000] == wanted[..], "what the flight wrote, round {round}");
            // (The copy over the first one's bytes came last.)
            let mut over = wanted[1..1501].to_vec();
            over[200..264].fill(0);
            assert!(title[HALF + 100..HALF + 1600] == over[..], "the copy to the other buffer, round {round}");
            assert!(title[1_500_000..1_500_700] == wanted[501..1201], "the copy behind a copy, round {round}");
            // SAFETY: the mapping, which the flight wrote and has landed.
            let staged = unsafe { std::slice::from_raw_parts(staged, 500) };
            assert!(staged == &wanted[601..1101], "the fill behind it, round {round}");
            assert!(title[900_001..900_778] == wanted[400..1177], "the copy within the buffer, round {round}");
            assert!(title[1_200_000..1_201_500] == over[..], "the copy of the copy, round {round}");
            assert!(title[HALF..HALF + 100].iter().all(|&x| x == 0));
            assert!(title[HALF + 1600..HALF + 10_000].iter().all(|&x| x == 0));
            assert!(title[HALF + 18_000..].iter().all(|&x| x == 0));
            let runs = &title[HALF + 10_000..HALF + 18_000];
            assert!(runs.chunks(2).all(|b| b == [1 + round as u8, 0]), "the runs, round {round}");
            a.keep(&upload);
            lists.extend([a, b, c]);
        }
        assert_eq!(host.log_messages(), None);
        let done = DONE[1].load(Ordering::Relaxed) - before;
        eprintln!("{done} of 48 copies were done on the card, {} submissions waited", DONE[3].load(Ordering::Relaxed));
        eprintln!("settles for title memory under shader writes: {}", LATE_STATS[WHY_STALE].load(Ordering::Relaxed));
        assert!(done >= 20, "only {done} copies were done on the card");
    }
}
