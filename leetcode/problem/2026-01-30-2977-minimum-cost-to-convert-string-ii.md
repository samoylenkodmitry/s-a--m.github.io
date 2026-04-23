---
layout: leetcode-entry
title: "2977. Minimum Cost to Convert String II"
permalink: "/leetcode/problem/2026-01-30-2977-minimum-cost-to-convert-string-ii/"
leetcode_ui: true
entry_slug: "2026-01-30-2977-minimum-cost-to-convert-string-ii"
---

[2977. Minimum Cost to Convert String II](https://leetcode.com/problems/minimum-cost-to-convert-string-ii/description/) hard
[blog post](https://leetcode.com/problems/minimum-cost-to-convert-string-ii/solutions/7536325/kotlin-by-samoylenkodmitry-xicf/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30012026-2977-minimum-cost-to-convert?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/nh9hqzaBT5w)

![c1fd649a-b247-4377-a4ae-b999f37dfc6b (1).webp](/assets/leetcode_daily_images/25e4185f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1253

#### Problem TLDR

Min cost to convert string with transitions of substrings #hard #dp #trie #hash #floyd-warshall

#### Intuition

1. Build transitions matrix of tokens
2. Do Floyd-Warshall for k i j  ij = min(ij, ik+kj)
3. Do DP to optimally split into substrings
4. Do rolling hashing to do substring in O(1) ammortized

```j
    //
    // TLE
    //
    // should i prepare all substrings?
    //
    // TLE dp as array?
    //
    // TLE
    //
    // expect me do write bottom up dp?
    // comments says Floyd-Warshall causes tle
    //
    // 100^3 = 1000000 should be in acceptable range...
    //
    // i'll try to write bottom-up then will gave up, don't want to write dijkstra
    //
    // so the substring calculation is what gaves me tle, its o(n^3), n = 1000
    //
```

#### Approach

* or do a Trie instead of rolling hashes, but we have to build transitions u,v from String to Trie id

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin
// 1231ms
    fun minimumCost(s: String, t: String, o: Array<String>, c: Array<String>, ct: IntArray): Long {
        val sa=o.toSet() + c; val a = sa.toList(); val ai = a.indices.associate { a[it] to it }
        val m = Array(a.size) { i -> LongArray(a.size) { if (i==it) 0L else 1L shl 60 }}
        for (i in o.indices) m[ai[o[i]]!!][ai[c[i]]!!] = min(1L*ct[i], m[ai[o[i]]!!][ai[c[i]]!!])
        for (k in a.indices) for (i in a.indices) for (j in a.indices) m[i][j]=min(m[i][j], m[i][k] + m[k][j])
        val dp = LongArray(s.length+1); val hs = a.map { it.fold(0) {h,c -> h*31+c.code }}.toSet()
        for (i in s.length-1 downTo 0) {
            dp[i] = if (s[i] == t[i]) dp[i + 1] else 1L shl 60; var sh = 0; var ht = 0
            for (j in i..<s.length) {
                sh = sh*31+s[j].code; ht = ht*31+t[j].code; if (sh !in hs || ht !in hs) continue
                val si = ai[s.substring(i,j+1)]?:continue; val ti = ai[t.substring(i,j+1)]?:continue
                dp[i] = min(dp[i], m[si][ti] + dp[j + 1])
            }
        }
        return if (dp[0] < 1L shl 60) dp[0] else -1L
    }
```
```kotlin
// 464ms
    fun minimumCost(s: String, t: String, o: Array<String>, c: Array<String>, ct: IntArray): Long {
        class T(var id: Int = -1) : HashMap<Char, T>(); val r = T(); var C = 0; val inf = 1L shl 60
        fun a(w: String) = w.fold(r) { n, c -> n.getOrPut(c, ::T) }.run { if (id < 0) id = C++; id }
        val u = o.map(::a); val v = c.map(::a); val m = Array(C) { i -> LongArray(C) { if (i == it) 0 else inf } }
        for (i in o.indices) m[u[i]][v[i]] = min(m[u[i]][v[i]], ct[i].toLong())
        for (k in 0..<C) for (i in 0..<C) for (j in 0..<C) m[i][j] = min(m[i][j], m[i][k] + m[k][j])
        val dp = LongArray(s.length + 1) { inf }; dp[0] = 0
        for (i in s.indices) {
            if (dp[i] >= inf) continue; if (s[i] == t[i]) dp[i + 1] = min(dp[i + 1], dp[i]); var x = r; var y = r
            for (j in i..<s.length) {
                x = x[s[j]]?:break; y = y[t[j]]?:break
                if (x.id >= 0 && y.id >= 0)  dp[j + 1] = min(dp[j + 1], dp[i] + m[x.id][y.id])
            }
        }
        return dp.last().takeIf { it < inf } ?: -1L
    }
```

