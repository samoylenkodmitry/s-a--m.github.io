---
layout: leetcode-entry
title: "3225. Maximum Score From Grid Operations"
permalink: "/leetcode/problem/2026-04-29-3225-maximum-score-from-grid-operations/"
leetcode_ui: true
entry_slug: "2026-04-29-3225-maximum-score-from-grid-operations"
---

[3225. Maximum Score From Grid Operations](https://leetcode.com/problems/maximum-score-from-grid-operations/solutions/8111758/kotlin-by-samoylenkodmitry-cwga/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29042026-3225-maximum-score-from?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6039xI26gpI)

https://dmitrysamoylenko.com/leetcode/

![29.04.2026.webp](/assets/leetcode_daily_images/29.04.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1343

#### Problem TLDR

Max sum adjacent to marked columns

#### Intuition

Didn't solve.
```j
    // looks like DP
    // at each column peek the best index, n^2 args and n inside = n^3
    // how to add previous white when doing current black and not overcount
    // 17 minute: my result is too much / too little
    //            meaning: my dp cases are wrong
    // 1:18: TLE, O(n^4) solution
    //
```
The O(n^4) idea: consider previous-previous(PP), previous(P) and the current (C) columns. Peek the best C to maximize sum of previous.
Use prefix sums of columns, result = max(PS(max(C,PP)) - PS(P))
The O(n^3) idea jump from n^4: submit to hard rule: can we take values from the current column or should we skip them. That allows to drop PP.

#### Approach

* the idea jump is not obvious, requires graphic visualizaition of possible outcomes

#### Complexity

- Time complexity:
$$O(n^3)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin
    fun maximumScore(g: Array<IntArray>): Long {
        val dp = HashMap<Int, Long>(); val ps = Array(g.size+1){LongArray(g.size+1)}
        for (x in g.indices) for (y in 1..g.size) ps[x+1][y] += ps[x+1][y-1]+g[y-1][x]
        fun dfs(i: Int, p: Int, skip: Int):Long = if (i==g.size)0L else dp.getOrPut(i*400+p*2+skip) {
            (0..g.size).maxOf { j ->
                val a = dfs(i+1, j, 0); val b = dfs(i+1, j, 1)
                if (skip > 0 && j > p) ps[i][j]-ps[i][p] + max(a, b)
                else if (skip < 1 && j <= p) max(ps[i+1][p]-ps[i+1][j] + a, b) else 0L
        }}
        return max(dfs(0, 0, 0),dfs(0,0,1))
    }
```
```rust

```

