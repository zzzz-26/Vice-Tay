/// Natural loops of the reachable, reducible CFG produced by `ir::build`.
pub(super) fn natural_loops(succ: &[Vec<usize>]) -> Vec<Vec<bool>> {
    let n = succ.len();
    if n == 0 {
        return Vec::new();
    }
    let (mut seen, mut post, mut stack) = (vec![false; n], Vec::new(), vec![(0, 0)]);
    seen[0] = true;
    while let Some((b, edge)) = stack.last_mut() {
        let from = *b;
        match succ[from].get(*edge) {
            Some(&to) => {
                *edge += 1;
                if !std::mem::replace(&mut seen[to], true) {
                    stack.push((to, 0));
                }
            }
            None => {
                post.push(from);
                stack.pop();
            }
        }
    }
    let mut rpo = vec![usize::MAX; n];
    for (rank, b) in post.into_iter().rev().enumerate() {
        rpo[b] = rank;
    }
    let mut pred = vec![Vec::new(); n];
    for (b, edges) in succ.iter().enumerate().filter(|(b, _)| seen[*b]) {
        for &t in edges {
            pred[t].push(b);
        }
    }
    let mut loops = std::collections::BTreeMap::new();
    for (tail, edges) in succ.iter().enumerate().filter(|(b, _)| seen[*b]) {
        for &header in edges.iter().filter(|&&t| rpo[t] <= rpo[tail]) {
            let body = loops.entry(header).or_insert_with(|| {
                let mut body = vec![false; n];
                body[header] = true;
                body
            });
            let mut pending = vec![tail];
            while let Some(b) = pending.pop() {
                if !std::mem::replace(&mut body[b], true) {
                    pending.extend(&pred[b]);
                }
            }
        }
    }
    loops.into_values().collect()
}
