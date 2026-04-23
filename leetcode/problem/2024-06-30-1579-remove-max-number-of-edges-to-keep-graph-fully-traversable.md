---
layout: leetcode-entry
title: "1579. Remove Max Number of Edges to Keep Graph Fully Traversable"
permalink: "/leetcode/problem/2024-06-30-1579-remove-max-number-of-edges-to-keep-graph-fully-traversable/"
leetcode_ui: true
entry_slug: "2024-06-30-1579-remove-max-number-of-edges-to-keep-graph-fully-traversable"
---

[1579. Remove Max Number of Edges to Keep Graph Fully Traversable](https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/description/) medium
[blog post](https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/solutions/5390506/kotiln-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30062024-1579-remove-max-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/jTf58qCm1-U)
![2024-06-30_11-27.webp](/assets/leetcode_daily_images/9ab1f681.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/655

#### Problem TLDR

Remove extra nodes in a connected graph by type 1, 2 and 3=1+2 #hard #union-find

#### Intuition

Type 3 nodes are the most valueable, let's keep them. Then check if type 1 is already connected by type 3 and do the same for type 2. To check connections use the Union-Find.

#### Approach

* at the end we can check connections to the first node, or just simple count how many edges added and compare it to n - 1
* both type1 and type2 must have add (n - 1) edges
* optimized Union-Find must have path compression and ranking, making time complexity O(1) (google inverse Akkerman function)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maxNumEdgesToRemove(n: Int, edges: Array<IntArray>): Int {
        class Uf(var v: Int = n - 1): HashMap<Int, Int>() {
            fun u(a: Int, b: Int): Boolean = if (f(a) == f(b)) false
                else { set(f(a), f(b)); v--; true }
            fun f(a: Int): Int = if (get(a) == a) a else
                get(a)?.let { b -> f(b).also { set(b, it) } } ?: a
        }
        val uu = List(2) { Uf() }; val u = Uf(); var count = 0
        for ((t, a, b) in edges) if (t == 3)
            if (!u.u(a, b)) count++ else for (u in uu) u.u(a, b)
        for ((t, a, b) in edges) if (t < 3) if (!uu[t - 1].u(a, b)) count++
        return if (uu.all { it.v < 1 }) count else -1
    }

```
```rust

    pub fn max_num_edges_to_remove(n: i32, edges: Vec<Vec<i32>>) -> i32 {
        fn u(uf: &mut Vec<usize>, a: &[i32]) -> i32 {
            let (fa, fb) = (f(uf, a[1] as usize), f(uf, a[2] as usize));
            if fa == fb { 0 } else { uf[fa] = fb; 1 }}
        fn f(uf: &mut Vec<usize>, a: usize) -> usize {
            let mut x = a; while x != uf[x] { x = uf[x] }; uf[a] = x; x }
        let mut u3 = (0..=n as usize).collect::<Vec<_>>();
        let (mut uu, mut v, mut res) = ([u3.clone(), u3.clone()], 2 * n - 2, 0);
        for e in &edges { if e[0] == 3 {
            if u(&mut u3, e) < 1 { res += 1 }
            else { for t in 0..2 { v -= u(&mut uu[t], e) }}}}
        for e in &edges { if e[0] < 3 {
            if u(&mut uu[e[0] as usize - 1], e) < 1 { res += 1 } else { v -= 1 }}}
        if v < 1 { res } else { -1 }
    }

```

