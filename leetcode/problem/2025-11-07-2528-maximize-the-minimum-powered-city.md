---
layout: leetcode-entry
title: "2528. Maximize the Minimum Powered City"
permalink: "/leetcode/problem/2025-11-07-2528-maximize-the-minimum-powered-city/"
leetcode_ui: true
entry_slug: "2025-11-07-2528-maximize-the-minimum-powered-city"
---

[2528. Maximize the Minimum Powered City](https://leetcode.com/problems/maximize-the-minimum-powered-city/description) hard
[blog post](https://leetcode.com/problems/maximize-the-minimum-powered-city/solutions/7332319/kotlin-by-samoylenkodmitry-68kz/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07112025-2528-maximize-the-minimum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/hschfrTcxKg)

![625a3497-7986-4f81-9162-7dedabcaa33c (1).webp](/assets/leetcode_daily_images/6e68e810.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1166

#### Problem TLDR

Max min r-range sum after adding total of k #hard #bs #sliding_window

#### Intuition

Define a target minimum range and binary search it.
To check if each value can add up to range in total of k budget use a sliding window.
Add to the rightmost position.

```j
    // 1 2 4 5 0    r=1  k=2
    // *         1+2   3 +2
    //   *       1+2+4 7
    //     *     2+4+5 11
    //       *   4+5+0 9
    //         * 5+0   5
    // the greedy idea: take the lowest city, but how many to add?
    // the k is 10^9, adding by constant can be too much steps
    // also adding to city propagates to range, so it O(adds*range)
    //
    // binary search idea: define the target minimum, greedily add up to it
    // how to add to a city in a linear way?
    //
    // we can have two ranges and a single optimal spot
    //
    //     [ range1 ]
    //            [ ]        the optimal spot
    //            [ range2 ]
    //
    // anyway, if we see the first city that needs more power add lazily
    //            [i + range - 1] += 1

    // to calculate powers we need O(n)
    // 1 1 1 1 1 1
    // [ 3 ]
    //   [ x ]
    //   x =  prev+1-1
```

#### Approach

* use a separate long array to keep track of the additions
*

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 89ms
    fun maxPower(s: IntArray, R: Int, k: Int): Long {
        var lo = 0L; var hi = Long.MAX_VALUE
        while (lo <= hi) {
            val m = lo + (hi - lo) / 2; var good = true
            val s = LongArray(s.size) { s[it].toLong() }
            var sum = 0L; var k = k.toLong(); var r = 0
            for (i in s.indices) {
                while (r < s.size && r - i <= R) sum += s[r++]
                if (sum < m) {
                    s[min(s.lastIndex,i+R)] += m - sum
                    k -= m - sum; sum = m
                }
                if (i - R >= 0) sum -= s[i-R]
                if (k < 0) { good = false; break }
            }
            if (good) lo = m + 1 else hi = m - 1
        }
        return hi
    }
```

