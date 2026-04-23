---
layout: leetcode-entry
title: "209. Minimum Size Subarray Sum"
permalink: "/leetcode/problem/2023-07-06-209-minimum-size-subarray-sum/"
leetcode_ui: true
entry_slug: "2023-07-06-209-minimum-size-subarray-sum"
---

[209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/description/) medium
[blog post](https://leetcode.com/problems/minimum-size-subarray-sum/solutions/3724899/kotlin-two-pointers/)
[substack](https://dmitriisamoilenko.substack.com/p/6072023-209-minimum-size-subarray?sd=pf)
![image.png](/assets/leetcode_daily_images/5a813ef1.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/267
#### Problem TLDR
Min length subarray with `sum >= target`
#### Intuition
Use two pointers: one adding to `sum` and another subtracting. As all numbers are positive, then `sum` will always be increasing with adding a number and deceasing when subtracting.

#### Approach
Let's use Kotlin `Sequence` API

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun minSubArrayLen(target: Int, nums: IntArray): Int {
    var lo = 0
    var sum = 0
    return nums.asSequence().mapIndexed { hi, n ->
        sum += n
        while (sum - nums[lo] >= target) sum -= nums[lo++]
        (hi - lo + 1).takeIf { sum >= target }
    }
    .filterNotNull()
    .min() ?: 0
}

```

