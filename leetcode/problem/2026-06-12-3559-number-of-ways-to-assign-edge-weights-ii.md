---
layout: leetcode-entry
title: "3559. Number of Ways to Assign Edge Weights II"
permalink: "/leetcode/problem/2026-06-12-3559-number-of-ways-to-assign-edge-weights-ii/"
leetcode_ui: true
entry_slug: "2026-06-12-3559-number-of-ways-to-assign-edge-weights-ii"
---

[3559. Number of Ways to Assign Edge Weights II](https://leetcode.com/problems/number-of-ways-to-assign-edge-weights-ii/solutions/8329365/kotlin-rust-by-samoylenkodmitry-2elc/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12062026-3559-number-of-ways-to-assign?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EwfsrTTu_NU)

https://dmitrysamoylenko.com/leetcode/

![12.06.2026.webp](/assets/leetcode_daily_images/12.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1388

#### Problem TLDR

Ways to color the all queries paths a..b in tree

#### Intuition

Didn't solved myself
```j
    // how to know path from each to each in less than O(n^2)?
    // use hints: LCA, DP of parity
    // TLE, probably because of my LCA
```

* construct the tree
* DFS: mark time for enter / exit for each node, later use to compare times to check for ancestor
* BFS: track depth of each node, later go to parent until depth is smaller
* the number of two-color the path is 2^len
* binary lifting: go up by powers of two jumps, this allows to precompute up[x][i]=up[up[x][i-1]][i-1], because i-1 is a half-jump https://cp-algorithms.com/graph/lca_binary_lifting.html

#### Approach

* don't forget to precompute powers of two

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(nlogn)$$

#### Code

```kotlin
    fun assignEdgeWeights(e: Array<IntArray>, q: Array<IntArray>) = run {
        var t=0; val n=e.size+2; val inT=IntArray(n); val d=IntArray(n)
        val p=(1..n).runningFold(1L){ r,_->(r*2)%1000000007L }
        val out = IntArray(n); out[0]=9999999; val up=Array(n){ IntArray(20){1} }
        val g=e.flatMap{(a,b)->setOf(a to b,b to a)}.groupBy({it.first},{it.second})
        fun anc(a:Int, b:Int) = inT[a]<=inT[b] && out[a]>=out[b]
        fun dfs(x:Int, p:Int) {
            inT[x]=++t; up[x][0]=p; for(i in 1..19) up[x][i] = up[up[x][i-1]][i-1]
            g[x]?.forEach{ if(it!=p) { d[it]=d[x]+1; dfs(it,x) } }; out[x]=++t
        }
        dfs(1,0); q.map { (a, b) ->
            var v=a; for(i in 19 downTo 0) if(!anc(up[v][i],b)) v=up[v][i]
            val x = if(anc(a,b)) a else if(anc(b,a)) b else up[v][0]
            val l = d[a]+d[b] - 2*d[x]; if(l==0) 0L else p[l-1]
        }
    }
```
```rust
    pub fn assign_edge_weights(e: Vec<Vec<i32>>, q: Vec<Vec<i32>>) -> Vec<i32> {
        let n = e.len() + 2; let mut p = vec![1; n];
        for i in 1..n { p[i] = p[i-1] * 2 % 1000000007; }
        let (mut d, mut up) = (vec![0; n], vec![[1; 20]; n]);
        let g = e.iter().flat_map(|v| { let (a,b)=(v[0]as usize,v[1]as usize); [(a, b), (b, a)] }).into_group_map();
        let (mut bfs, mut i) = (vec![1], 0);
        while i < bfs.len() {
            let x = bfs[i]; i += 1;
            for j in 1..20 { up[x][j] = up[up[x][j-1]][j-1]; }
            for &v in &g[&x] { if v != up[x][0] { d[v]=d[x]+1; up[v][0]=x; bfs.push(v); } }
        }
        q.iter().map(|v| {
            let (mut a, mut b) = (v[0] as usize, v[1] as usize);
            let (da, db) = (d[a], d[b]);
            if d[a] < d[b] { std::mem::swap(&mut a, &mut b); }
            for j in 0..20 { if ((d[a] - d[b]) >> j) & 1 == 1 { a = up[a][j]; } }
            let x = if a == b { a } else {
                for j in (0..20).rev() { if up[a][j] != up[b][j] { a=up[a][j]; b=up[b][j]; } }
                up[a][0]
            };
            let l = da + db - 2 * d[x]; if l == 0 { 0 } else { p[l - 1] }
        }).collect()
    }
```

