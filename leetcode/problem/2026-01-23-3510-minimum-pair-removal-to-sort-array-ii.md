---
layout: leetcode-entry
title: "3510. Minimum Pair Removal to Sort Array II"
permalink: "/leetcode/problem/2026-01-23-3510-minimum-pair-removal-to-sort-array-ii/"
leetcode_ui: true
entry_slug: "2026-01-23-3510-minimum-pair-removal-to-sort-array-ii"
---

[3510. Minimum Pair Removal to Sort Array II](https://leetcode.com/problems/minimum-pair-removal-to-sort-array-ii/description/) hard
[blog post](https://leetcode.com/problems/minimum-pair-removal-to-sort-array-ii/solutions/7517576/kotlin-by-samoylenkodmitry-wfql/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23012026-3510-minimum-pair-removal?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/IwcuKlvWyvc)

![2aaada48-9d5c-422c-a4e9-fb4df9ce90bc (1).webp](/assets/leetcode_daily_images/589e53a5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1246

#### Problem TLDR

Sort by removing min-sum pairs #hard

#### Intuition

Didn't solve.
```j
    // seen it yesterday
    // it's too hard
    // i'll try to do it from memory
    //
    // the algo:
    // make a linked list
    // put pairs into sorted heap
    // remove one by one
    //
    //       LL L i R RR
    //              *
    //              remove
    // i gave up
```

* put sums into heap
* poll and remove right value of the pair
* adjust count of unordered pairs before the removal and after

#### Approach

* careful with overflow, can't put everything in Long
* sentinels at both sides will help to avoid some checks

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 805ms
    fun minimumPairRemoval(n: IntArray): Int {
        val A = LongArray(n.size) {1L*n[it]} + Long.MAX_VALUE/2
        val q = PriorityQueue<Pair<Long,Int>>(compareBy({it.first},{it.second}))
        val L = IntArray(A.size) { it-1 }; val R = IntArray(A.size) { it+1 }
        q += (0..<n.size).map {(A[it]+A[it+1]) to it}; var res = 0
        fun b(i: Int, j: Int) = if (i>=0&&A[i] > A[j]) 1 else 0
        var c = (0..<n.size).sumOf{b(it,it+1)}
        while (c > 0) {
            val (s,i) = q.poll(); if (L[R[i]] != i || s != A[i]+A[R[i]]) continue
            c -= b(L[i], i) + b(i, R[i]) + b(R[i], R[R[i]])
            A[i] = s; R[i] = R[R[i]]; L[R[i]] = i
            c += b(L[i], i) + b(i, R[i])
            if (L[i] >= 0) q += (1L*s+A[L[i]]) to L[i]
            if (R[i] <= n.size) q += (1L*s+A[R[i]]) to i
            res++
        }
        return res
    }
```

