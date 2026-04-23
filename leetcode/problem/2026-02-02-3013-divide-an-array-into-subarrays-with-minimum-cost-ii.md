---
layout: leetcode-entry
title: "3013. Divide an Array Into Subarrays With Minimum Cost II"
permalink: "/leetcode/problem/2026-02-02-3013-divide-an-array-into-subarrays-with-minimum-cost-ii/"
leetcode_ui: true
entry_slug: "2026-02-02-3013-divide-an-array-into-subarrays-with-minimum-cost-ii"
---

[3013. Divide an Array Into Subarrays With Minimum Cost II](https://leetcode.com/problems/divide-an-array-into-subarrays-with-minimum-cost-ii/description) hard
[blog post](https://leetcode.com/problems/divide-an-array-into-subarrays-with-minimum-cost-ii/solutions/7545399/kotlin-by-samoylenkodmitry-zi3v/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02022026-3013-divide-an-array-into?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/yeZWsowld9E)

![e4bff72e-a552-41a0-b499-e1adf5145668 (1).webp](/assets/leetcode_daily_images/fb3ca13f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1256

#### Problem TLDR

Min sum of k splits at max d distance #hard #sliding_window #heap

#### Intuition

Didn't solve.
```j
    // take first
    // select k-1 min-sum values at dist
    //
    // ***********
    //   * * *
    //
    // move (k-1) pointers?
    //
    // binarysearch?
    //
    // binarysearch+dp? dp answers for sum: canSplit[i]
    //                         still should be minSplit[i] O(n^2)
    // no other ideas, lets try dp, inner cycle is d, O(nkd)
    //                       obviously deadend tle
    // somehow stuck with brute-force dp
    //   looks like i didn't understood the description
    //       dist is not between splits
    //                it is between second and last
    //
    // let's go to hints
    //
    // sliding window + heap + heap
    //
    // *|**m*|********
    //  i-d  i
    //
    // let's give up at 50 minutes
```

1. Maintain sliding window of d
2. Put k-1 best values in one sorted container (TreeSet of n[i],i)
3. Remove overflows and d+1-distant values.
4. Keep still-in-window values in a second storage container
5. Balance when d+1-distant value has been removed

#### Approach

* there is also a BIT solution (ask ai): sort distinct values, keep BIT-arrays of counts and sums

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 533ms
    fun minimumCost(n: IntArray, k: Int, d: Int): Long {
        var sum = 0L; val c = compareBy<Int>({n[it]},{it})
        val q = TreeSet(c); val s = TreeSet(c)
        return (1..<n.size).minOf { i ->
            q += i; sum += n[i]
            if (q.size >= k) { val j = q.pollLast(); sum -= n[j]; s += j }
            if (i-d-1 > 0 && !s.remove(i-d-1) && q.remove(i-d-1)) {
                sum -= n[i-d-1]; val j = s.pollFirst(); sum += n[j]; q += j
            }
            if (i-d-1 >= 0) sum else 1L shl 60
        } + n[0]
    }
```

