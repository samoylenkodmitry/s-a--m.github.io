---
layout: leetcode-entry
title: "3841. Palindromic Path Queries in a Tree"
permalink: "/leetcode/problem/2026-10-29-3841-palindromic-path-queries-in-a-tree/"
leetcode_ui: true
entry_slug: "2026-10-29-3841-palindromic-path-queries-in-a-tree"
---

[3841. Palindromic Path Queries in a Tree](https://leetcode.com/problems/palindromic-path-queries-in-a-tree/solutions/8593618/kotlin-by-samoylenkodmitry-lgmj/) hard
[substack](https://dmitriisamoilenko.substack.com/p/29102026-3841-palindromic-path-queries?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/CxKZqJaX_v4)

https://dmitrysamoylenko.com/leetcode/

![29.10.2026.webp](/assets/leetcode_daily_images/29.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1526

#### Problem TLDR

Queries of update value, query path u,v can be palindrome in a tree

#### Intuition

Didn't tried to solve.
Subproblems:
1. palindrome - xor of path = 1 means it can be palindrome
2. tracking ancestors - Euler tour DFS marking time in and time out of the node answers is ancestor by comparing the times
3. prefix xors from root to both nodes xor the value of Lowest Common Ancestor gives a xor of the path between the node, common prefix self-destructs on xor
4. store prefixes in Fenwick Tree by nodes time; query i-=i&-i; modify i+=i&-i
5. find LCA with jump table Binary Lifting: build 0->16 up[0][v]=v; up[k][v]=up[k-1][up[k-1][v]], query 16->0 if jump is not yet ancestor do the jump, v stays, u jumps

#### Approach

* don't give up

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(nlong)$$

#### Code

```kotlin
    fun palindromePath(n: Int, e: Array<IntArray>, s: String, q: Array<String>)=run{
        val g = Array(n) { ArrayList<Int>() }; for ((u, v) in e) { g[u]+=v; g[v] +=u}
        val up = Array(17) { IntArray(n) }
        val tin = IntArray(n); val tout = IntArray(n); val bit = IntArray(n + 2)
        val mask = IntArray(n) { 1 shl (s[it] - 'a') }; var t = 0
        fun add(i: Int, v: Int) { if (i <= n) { bit[i] = bit[i] xor v; add(i + (i and -i), v) } }
        fun sum(i: Int, r: Int = 0): Int = if (i == 0) r else sum(i - (i and -i), r xor bit[i])
        fun upd(u: Int, v: Int) { add(tin[u], v); add(tout[u] + 1, v) }
        fun dfs(u: Int) {
            if (tin[u] == 0) tin[u] = ++t
            val v = g[u].removeLastOrNull()
            if (v != null) {
                if (v != up[0][u]) {
                    up[0][v] = u
                    for (k in 1..16) up[k][v] = up[k - 1][up[k - 1][v]]
                    return dfs(v)
                }
                return dfs(u)
            }
            tout[u] = t; upd(u, mask[u])
            if (u != 0) dfs(up[0][u])
        }
        dfs(0)
        fun anc(u: Int, v: Int) = tin[u] <= tin[v] && tout[v] <= tout[u]
        fun lca(u0: Int, v: Int): Int {
            var u = u0
            if (anc(u, v)) return u
            for (k in 16 downTo 0) if (!anc(up[k][u], v)) u = up[k][u]
            return up[0][u]
        }
        q.mapNotNull { str ->
            val p = str.split(' '); val u = p[1].toInt()
            if (p[0][0] == 'u') {
                (1 shl (p[2][0] - 'a')).let { upd(u, mask[u] xor it); mask[u] = it; null }
            } else {
                val v = p[2].toInt()
                (sum(tin[u]) xor sum(tin[v]) xor mask[lca(u, v)]).countOneBits() <= 1
            }
        }
    }
```
```rust

```

