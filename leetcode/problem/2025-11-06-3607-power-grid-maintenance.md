---
layout: leetcode-entry
title: "3607. Power Grid Maintenance"
permalink: "/leetcode/problem/2025-11-06-3607-power-grid-maintenance/"
leetcode_ui: true
entry_slug: "2025-11-06-3607-power-grid-maintenance"
---

[3607. Power Grid Maintenance](https://leetcode.com/problems/power-grid-maintenance/description) medium
[blog post](https://leetcode.com/problems/power-grid-maintenance/solutions/7330347/kotlin-rust-by-samoylenkodmitry-3h2e/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06112025-3607-power-grid-maintenance?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/LcWB-VLkcDI)

![4273c241-4875-4ebd-ac94-b83dfcb1f7fb (1).webp](/assets/leetcode_daily_images/f8f9e107.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1165

#### Problem TLDR

Smallest in subgraph after turning some nodes off #medium #uf

#### Intuition

Union-Find to build subgraphs.
Group by roots and put into TreeSets.

#### Approach

* use Map<Root,TreeSet>

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 223ms
    fun processQueries(c: Int, conn: Array<IntArray>, q: Array<IntArray>)=buildList<Int>{
        val u = IntArray(c+1) { it }
        fun f(x: Int): Int = if (x == u[x]) x else f(u[x]).also {u[x] = it}
        for ((a,b) in conn) u[f(a)] = f(b)
        val g = (1..c).groupBy { f(it) }.mapValues {TreeSet<Int>(it.value)}
        for ((t, v) in q) g[f(v)]?.let {if (t == 2) it.remove(v) else
            add(if (v in it) v else it.firstOrNull()?:-1) }
    }

```
```rust
// 99ms
    pub fn process_queries(c: i32, conn: Vec<Vec<i32>>, q: Vec<Vec<i32>>) -> Vec<i32> {
        let mut u: Vec<_> = (0..=c as usize).collect();
        fn f(u: &mut Vec<usize>, x: usize) -> usize { if x==u[x] { x } else { let r = f(u,u[x]); u[x]=r; r} }
        for e in conn { let (a,b) = (e[0]as usize,e[1]as usize); let r = f(&mut u,a); u[r]=f(&mut u,b);}
        let mut g = HashMap::new();
        for i in 1..=c as usize {g.entry(f(&mut u,i)).or_insert_with(BTreeSet::new).insert(i as i32);}
        q.iter().map(|p|{
            let (t,v) = (p[0],p[1]); let s = g.get_mut(&f(&mut u,v as usize)).unwrap();
            if t > 1 { s.remove(&v);-2} else { if s.contains(&v) { v} else { *s.iter().next().unwrap_or(&-1)}}
        }).filter(|&r| r > -2).collect()
    }

```

