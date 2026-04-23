---
layout: leetcode-entry
title: "239. Sliding Window Maximum"
permalink: "/leetcode/problem/2023-08-16-239-sliding-window-maximum/"
leetcode_ui: true
entry_slug: "2023-08-16-239-sliding-window-maximum"
---

[239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/description/) medium
[blog post](https://leetcode.com/problems/sliding-window-maximum/solutions/3915981/kotlin-monotonic-queue/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16082023-239-sliding-window-maximum?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/fd8095c0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/310

#### Problem TLDR

List of sliding window's maximums

#### Intuition

To quickly find a maximum in a sliding window, consider example:

```
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Window position                Max
---------------               -----
[#  3  -1]  _  _  _  _  _       3
 _ [3  -1  -3] _  _  _  _       3
 _  _ [ #   #  5] _  _  _       5
 _  _   _ [ #  5  3] _  _       5
 _  _   _   _ [#  #  6] _       6
 _  _   _   _  _ [#  #  7]      7

```
After each new maximum appends to the end of the window, they become the maximum until the window slides it out, so all lesser positions to the left of it are irrelevant.

#### Approach

We can use a decreasing `Stack` technique to remove all the smaller elements. However, to maintain a window size, we'll need a `Queue`.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maxSlidingWindow(nums: IntArray, k: Int): IntArray = with(ArrayDeque<Int>()) {
        val res = mutableListOf<Int>()
        nums.forEachIndexed { i, x ->
          while (isNotEmpty() && nums[peekLast()] < x) removeLast()
          add(i)
          while (isNotEmpty() && i - peekFirst() + 1 > k) removeFirst()
          if (i >= k - 1) res += nums[peekFirst()]
        }
        return res.toIntArray()
    }

```

