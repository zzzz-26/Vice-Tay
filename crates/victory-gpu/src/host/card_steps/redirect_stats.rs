//! Placed-copy read-through observations at the final cut or preparation decision.
//! Counts are totals per reporting interval, separated by the switch value that
//! produced the decision. A stale or unused preplan contributes nothing.
//! Enable with the redirect A/B switch or `VICTORY_PLACED_FILL_REASONS=1`.
//! Planned fills belong to a consumed, accepted plan. Verified fills passed
//! the source and alias checks. Executed fills were recorded in the closed
//! host list, including consumers that share one GPU read. Promoted fills
//! are the subset added when preparation retries a blocked CPU fill.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use super::{Planned, RedirectWait};

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        crate::ab::bit("VICTORY_PLACED_FILL_REDIRECT").is_some()
            || crate::switch::on("VICTORY_PLACED_FILL_REASONS")
    })
}

#[derive(Default)]
struct Totals {
    planned: u64,
    promoted: u64,
    verified: u64,
    executed: u64,
    missing_target: u64,
    cuts: BTreeMap<&'static str, u64>,
    refusals: BTreeMap<&'static str, u64>,
    after_plan: BTreeMap<&'static str, u64>,
    fallback_waits: BTreeMap<&'static str, WaitTime>,
}

/// Wall time in the fallback for a submission still blocked after card_steps.
#[derive(Default, Debug)]
struct WaitTime {
    calls: u64,
    total_us: u64,
    max_us: u64,
}

fn totals() -> &'static Mutex<[Totals; 2]> {
    static TOTALS: OnceLock<Mutex<[Totals; 2]>> = OnceLock::new();
    TOTALS.get_or_init(|| Mutex::new(Default::default()))
}

pub(super) fn cut(on: bool, cause: &'static str) {
    if enabled() {
        let mut totals = totals().lock().unwrap_or_else(|e| e.into_inner());
        *totals[usize::from(on)].cuts.entry(cause).or_default() += 1;
    }
}

pub(super) fn wait(on: bool, cause: &'static str, elapsed: std::time::Duration) {
    let us = elapsed.as_micros() as u64;
    let mut totals = totals().lock().unwrap_or_else(|e| e.into_inner());
    let t = totals[usize::from(on)].fallback_waits.entry(cause).or_default();
    t.calls += 1;
    t.total_us += us;
    t.max_us = t.max_us.max(us);
}

/// One actual preparation attempt. The caller commits it once, including on failure.
pub(super) struct Attempt {
    on: bool,
    pub(super) cause: Option<&'static str>,
    planned: u64,
    pub(super) promoted: u64,
    pub(super) verified: u64,
    pub(super) executed: u64,
    pub(super) missing_target: u64,
}

impl Attempt {
    pub(super) fn new(on: bool) -> Self {
        Self { on, cause: None, planned: 0, promoted: 0, verified: 0, executed: 0, missing_target: 0 }
    }

    pub(super) fn planned(&mut self, steps: &[Planned]) {
        if enabled() {
            self.planned = steps.iter().filter(|p| p.redirect.is_some()).count() as u64;
        }
    }

    fn add_to(self, counts: &mut Totals, failure: Option<&'static str>) {
        counts.planned += self.planned;
        counts.promoted += self.promoted;
        counts.verified += self.verified;
        counts.executed += self.executed;
        counts.missing_target += self.missing_target;
        if let Some(cause) = self.cause {
            *counts.refusals.entry(cause).or_default() += 1;
        } else if self.planned > 0 && let Some(why) = failure {
            *counts.after_plan.entry(why).or_default() += 1;
        }
    }

    pub(super) fn finish(self, failure: Option<&'static str>) -> Option<RedirectWait> {
        if (self.cause.is_some() || self.planned > 0) && enabled() {
            let wait = failure.map(|why| RedirectWait { on: self.on, why: self.cause.unwrap_or(why) });
            let mut totals = totals().lock().unwrap_or_else(|e| e.into_inner());
            let counts = &mut totals[usize::from(self.on)];
            self.add_to(counts, failure);
            return wait;
        }
        None
    }
}

pub(super) fn line() {
    if !enabled() {
        return;
    }
    let both = std::mem::take(&mut *totals().lock().unwrap_or_else(|e| e.into_inner()));
    for (on, c) in [false, true].into_iter().zip(both) {
        if c.planned != 0 || !c.cuts.is_empty() || !c.refusals.is_empty() || !c.fallback_waits.is_empty() {
            tracing::info!(
                redirect = on, planned_fills = c.planned, promoted_planned_fills = c.promoted,
                verified_fills = c.verified,
                executed_fills = c.executed, missing_fill_target = c.missing_target,
                cut_reasons = ?c.cuts, refusal_reasons = ?c.refusals,
                failures_after_plan = ?c.after_plan, fallback_waits = ?c.fallback_waits,
                "placed fill redirect decisions"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_batch_keeps_its_planned_and_verified_fills() {
        let mut totals = Totals::default();
        let failed = Attempt { planned: 3, promoted: 2, verified: 1, ..Attempt::new(true) };
        failed.add_to(&mut totals, Some("mapping changed"));
        assert_eq!((totals.planned, totals.verified, totals.executed), (3, 1, 0));
        assert_eq!(totals.promoted, 2);
        assert_eq!(totals.after_plan["mapping changed"], 1);
        assert!(totals.refusals.is_empty());

        let mut refusal = Attempt::new(true);
        refusal.cause = Some("partial_coverage");
        refusal.add_to(&mut totals, Some(super::super::BEHIND_PLACED));
        assert_eq!(totals.refusals["partial_coverage"], 1);
        assert!(totals.cuts.is_empty());
        assert_eq!(totals.after_plan.len(), 1);
    }
}
