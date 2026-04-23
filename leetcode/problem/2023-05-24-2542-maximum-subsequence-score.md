---
layout: leetcode-entry
title: "2542. Maximum Subsequence Score"
permalink: "/leetcode/problem/2023-05-24-2542-maximum-subsequence-score/"
leetcode_ui: true
entry_slug: "2023-05-24-2542-maximum-subsequence-score"
---

[2542. Maximum Subsequence Score](https://leetcode.com/problems/maximum-subsequence-score/description/) medium
[blog post](https://leetcode.com/problems/maximum-subsequence-score/solutions/3557549/kotlin-priorityqueue/)
[substack](https://dmitriisamoilenko.substack.com/p/24052023-2542-maximum-subsequence?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/222
#### Problem TLDR
Max score of `k` sum(subsequence(a)) * min(subsequence(b))
#### Intuition
First, the result is independent of the order, so we can sort. For maximum score, it better to start with maximum multiplier of `min`. Then, we iterate from biggest nums2 to smallest. Greedily add numbers until we reach `k` elements. After `size > k`, we must consider what element to extract. Given our `min` is always the current value, we can safely take any element without modifying the minimum, thus take out the smallest by `nums1`.

#### Approach
* use `PriorityQueue` to dynamically take out the smallest
* careful to update score only when `size == k`, as it may decrease with more elements
#### Complexity
- Time complexity:
$$O(nlog(n))$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun maxScore(nums1: IntArray, nums2: IntArray, k: Int): Long {
    // 14  2 1 12 100000000000  1000000000000 100000000000
    // 13 11 7 1  1             1             1
    val inds = nums1.indices.sortedWith(
    compareByDescending<Int> { nums2[it] }
        .thenByDescending { nums1[it] })
    var score = 0L
    var sum = 0L
    val pq = PriorityQueue<Int>(compareBy { nums1[it] })
    inds.forEach {
        sum += nums1[it].toLong()
        pq.add(it)
        if (pq.size > k) sum -= nums1[pq.poll()].toLong()
        if (pq.size == k) score = maxOf(score, sum * nums2[it].toLong())
    }
    return score
}

```

