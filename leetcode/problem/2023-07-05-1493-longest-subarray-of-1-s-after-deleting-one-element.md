---
layout: leetcode-entry
title: "1493. Longest Subarray of 1's After Deleting One Element"
permalink: "/leetcode/problem/2023-07-05-1493-longest-subarray-of-1-s-after-deleting-one-element/"
leetcode_ui: true
entry_slug: "2023-07-05-1493-longest-subarray-of-1-s-after-deleting-one-element"
---

[1493. Longest Subarray of 1's After Deleting One Element](https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/description/) medium
[blog post](https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/solutions/3720190/kotlin-3-pointers/)
[substack](https://dmitriisamoilenko.substack.com/p/5072023-1493-longest-subarray-of?sd=pf)
![image.png](/assets/leetcode_daily_images/42e3f0f0.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/266
#### Problem TLDR
Largest `1..1` subarray after removing one item
#### Intuition
Let's maintain two pointers for a `start` and a `nextStart` positions, and a third pointer for the `right` border.

* move `start` to the `nextStart` when `right` == 0
* move `nextStart` to start of `1`'s

#### Approach
* corner case is when all array is `1`'s, as we must remove `1` then anyway

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ add `asSequence` for it to become $$O(1)$$

#### Code

```kotlin

fun longestSubarray(nums: IntArray): Int {
    var start = -1
    var nextStart = -1
    return if (nums.sum() == nums.size) nums.size - 1
    else nums.mapIndexed { i, n ->
        if (n == 0) {
            start = nextStart
            nextStart = -1
            0
        } else {
            if (nextStart == -1) nextStart = i
            if (start == -1) start = nextStart
            i - start + (if (start == nextStart) 1 else 0)
        }
    }.max() ?:0
}

```

