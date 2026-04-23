---
layout: leetcode-entry
title: "2090. K Radius Subarray Averages"
permalink: "/leetcode/problem/2023-06-20-2090-k-radius-subarray-averages/"
leetcode_ui: true
entry_slug: "2023-06-20-2090-k-radius-subarray-averages"
---

[2090. K Radius Subarray Averages](https://leetcode.com/problems/k-radius-subarray-averages/description/) medium
[blog post](https://leetcode.com/problems/k-radius-subarray-averages/solutions/3659377/kotlin-sliding-window/)
[substack](https://dmitriisamoilenko.substack.com/p/20062023-2090-k-radius-subarray-averages?sd=pf)
![image.png](/assets/leetcode_daily_images/9c84b246.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/251
#### Problem TLDR
Array containing sliding window of size `2k+1` average or `-1`
#### Intuition
Just do what is asked

#### Approach
* careful with `Int` overflow
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun getAverages(nums: IntArray, k: Int): IntArray {
    if (k == 0) return nums
    var sum = 0L
    val res = IntArray(nums.size) { -1 }
    for (i in 0 until nums.size) {
        sum += nums[i]
        if (i > 2 * k) sum -= nums[i - 2 * k - 1]
        if (i >= 2 * k) res[i - k] = (sum / (2 * k + 1)).toInt()
    }
    return res
}

```

