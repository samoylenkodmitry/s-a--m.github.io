---
layout: leetcode-entry
title: "3310. Remove Methods From Project"
permalink: "/leetcode/problem/2026-08-05-3310-remove-methods-from-project/"
leetcode_ui: true
entry_slug: "2026-08-05-3310-remove-methods-from-project"
---

[3310. Remove Methods From Project](https://leetcode.com/problems/remove-methods-from-project/solutions/8442207/kotlin-rust-by-samoylenkodmitry-osdz/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05082026-3310-remove-methods-from?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_vnbktX0IFI)

https://dmitrysamoylenko.com/leetcode/

![05.08.2026.webp](/assets/leetcode_daily_images/05.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1442

#### Problem TLDR

Remove isolated group started from K

#### Intuition

Collect nodes with DFS or BFS. Then check if subgraph has any incoming edge.

#### Approach

* instead of the last iteration we can count in/out edges for the visited group and it is isolated if in==out

#### Complexity

- Time complexity:
$$O(V+E)$$

- Space complexity:
$$O(V+E)$$

#### Code

```kotlin
    fun remainingMethods(n: Int, k: Int, inv: Array<IntArray>)=run {
        val g = inv.groupBy({it[0]},{it[1]}); val vis = hashSetOf(k)
        fun dfs(x: Int) { g[x]?.forEach {if (vis.add(it)) dfs(it) }}
        dfs(k)
        if (inv.any{(a,b)-> b in vis&& a !in vis}) vis.clear(); (0..<n)-vis
    }
```
```rust
    pub fn remaining_methods(n: i32, k: i32, inv: Vec<Vec<i32>>) -> Vec<i32> {
        let (g, mut vis, mut q) = (inv.iter().map(|e| (e[0], e[1])).into_group_map(), HashSet::from([k]), vec![k]);
        while let Some(u)=q.pop() { if let Some(l) = g.get(&u) { for &v in l { if vis.insert(v) { q.push(v) }}}}
        if inv.iter().any(|e| vis.contains(&e[1]) > vis.contains(&e[0])) { vis.clear() }
        (0..n).filter(|x| !vis.contains(x)).collect()
    }
```

