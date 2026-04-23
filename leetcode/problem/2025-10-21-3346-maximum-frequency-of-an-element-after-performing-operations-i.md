---
layout: leetcode-entry
title: "3346. Maximum Frequency of an Element After Performing Operations I"
permalink: "/leetcode/problem/2025-10-21-3346-maximum-frequency-of-an-element-after-performing-operations-i/"
leetcode_ui: true
entry_slug: "2025-10-21-3346-maximum-frequency-of-an-element-after-performing-operations-i"
---

[3346. Maximum Frequency of an Element After Performing Operations I](https://leetcode.com/problems/maximum-frequency-of-an-element-after-performing-operations-i/description) medium
[blog post](https://leetcode.com/problems/maximum-frequency-of-an-element-after-performing-operations-i/solutions/7290193/kotlin-by-samoylenkodmitry-ajrh/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21102025-3346-maximum-frequency-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/tykaR8cF-SM)

![00ba3e93-d803-4aee-b4c2-be4f275895f9 (1).webp](/assets/leetcode_daily_images/bc2bc8d8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1149

#### Problem TLDR

Max frequency after changing +k..-k ops elements #medium #sliding_window

#### Intuition

Didn't solved.

```j
    // 5 6 7 8 10 15 20        k=5
    // maybe binary search? for resulting number in lowest..highest
    //       no rule for bs
    // sliding window
    // how to track how many was changed?
    // group?
    // 0 0 0 0 5 5 10 10 10   k=5 o=2
    // 0 0 0 0 5 5 10 10 10   k=5 o=1
    // 0 0 5 5 5 5            k=5 o=1
    // 0 1 1 1 5 5            k=5 o=1
    // maintain frequency of each element in window, know max frequency,
    //   ans = min(max_frequency_in_window+ops, window_size)
    // for frequencies: running sorted window, keep map num:freq; when add sorted-f[num]+(++f[num])
    // looks complicated, maybe wrong
    // 1: window is 2*k
    // 2: can't just use most frequent
    // 35 minute: let's look for hints
    // try each as candidate (j)
    // how to count number of operations? window-curr_freq is not correct
    // 5 11 20 20 when current freq = 2 of '20'; to do the 2*k range we have to use 2 operations
    // and to do 1*k range we can use 0 operations
    // 1:14 failed on 58 80 5 let's give up
```

Solve two separate problems:
1. choose every number as baseline, window b-k..b+k
2. no baseline, just 2k window

#### Approach

* sometimes there is no single algorithm, just separate tasks for different cases

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 125ms
    fun maxFrequency(n: IntArray, k: Int, o: Int): Int {
        n.sort(); var res = 0; var l = 0; var r = 0; var f = HashMap<Int,Int>()
        for (x in n) {
            while (r < n.size && n[r] <= x+k) f[n[r]] = 1 + (f[n[r++]] ?: 0)
            while (l < n.size && n[l] < x-k) f[n[l]] = -1 + f[n[l++]]!!
            res = max(res, min(f[x]!!+o, r - l))
        }
        l = 0
        return max(res, n.indices.maxOf { r ->  while (n[r] - n[l] > 2*k) ++l; min(r-l+1, o) })
    }

```

