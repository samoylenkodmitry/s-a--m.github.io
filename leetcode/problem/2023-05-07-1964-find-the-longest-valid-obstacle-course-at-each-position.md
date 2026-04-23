---
layout: leetcode-entry
title: "1964. Find the Longest Valid Obstacle Course at Each Position"
permalink: "/leetcode/problem/2023-05-07-1964-find-the-longest-valid-obstacle-course-at-each-position/"
leetcode_ui: true
entry_slug: "2023-05-07-1964-find-the-longest-valid-obstacle-course-at-each-position"
---

[1964. Find the Longest Valid Obstacle Course at Each Position](https://leetcode.com/problems/find-the-longest-valid-obstacle-course-at-each-position/description/) hard

```kotlin

fun longestObstacleCourseAtEachPosition(obstacles: IntArray): IntArray {
    // 2 3 1 3
    // 2          2
    //   3        2 3
    //     1      1 3    (pos = 1)
    //       3    1 3 3

    // 5 2 5 4 1 1 1 5 3 1
    // 5       .             5
    //   2     .             2
    //     5   .             2 5
    //       4 .             2 4
    //         1             1 4 (pos = 1)
    //           1           1 1
    //             1         1 1 1
    //               5       1 1 1 5
    //                 3     1 1 1 3
    //                   1   1 1 1 1

    val lis = IntArray(obstacles.size)
    var end = 0
    return obstacles.map { x ->
        var pos = -1
        var lo = 0
        var hi = end - 1
        while (lo <= hi) {
            val mid = lo + (hi - lo) / 2
            if (lis[mid] > x) {
                hi = mid - 1
                pos = mid
            } else lo = mid + 1
        }
        if (pos == -1) {
            lis[end++] = x
            end
        } else {
            lis[pos] = x
            pos + 1
        }
    }.toIntArray()
}

```

[blog post](https://leetcode.com/problems/find-the-longest-valid-obstacle-course-at-each-position/solutions/3495432/kotlin-lis/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-7052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/205
#### Intuition
This is the [Longest Increasing Subsequence](https://en.wikipedia.org/wiki/Longest_increasing_subsequence) length problem, that have a classic algorithm, which must be learned and understood.

The trivial case of `any increasing subsequence` is broken by example: `2 3 1 3`, when we consider the last `3` result must be: `233` instead of `13`. So, we must track all the sequences.

To track all the sequences, we can use `TreeMap` that will hold the `largest` element and length of any subsequence. Adding a new element will take $$O(n^2)$$.

The optimal `LIS` solution is to keep the largest increasing subsequence so far and cleverly add new elements:
1. for a new element, search for the `smallest` element that is `larger` than it
2. if found, replace
3. if not, append
![lis.gif](/assets/leetcode_daily_images/314d1821.webp)

#### Approach
* google what is the solution of `LIS`
* use an array for `lis`
* carefully write binary search
#### Complexity
- Time complexity:
$$O(nlog(n))$$
- Space complexity:
$$O(n)$$

