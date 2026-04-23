---
layout: leetcode-entry
title: "1685. Sum of Absolute Differences in a Sorted Array"
permalink: "/leetcode/problem/2023-11-25-1685-sum-of-absolute-differences-in-a-sorted-array/"
leetcode_ui: true
entry_slug: "2023-11-25-1685-sum-of-absolute-differences-in-a-sorted-array"
---

[1685. Sum of Absolute Differences in a Sorted Array](https://leetcode.com/problems/sum-of-absolute-differences-in-a-sorted-array/description/) medium
[blog post](https://leetcode.com/problems/sum-of-absolute-differences-in-a-sorted-array/solutions/4326893/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25112023-1685-sum-of-absolute-differences?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/7bgwjpdKCD0)
![image.png](/assets/leetcode_daily_images/b3099ab9.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/416

#### Problem TLDR

Array to `sum_j(abs(arr[i] - arr[j]))` for each `i`

#### Intuition

This is an arithmetic problem. We need to pay attention of an `abs` sign, given the array in a sorted order.

```
  // 0 1 2 3 4 5
  // a b c d e f
  // c: c-a + c-b + c-c + d-c + e-c + f-c
  // c * (1 + 1 + 1-1 -1 -1-1) -a-b+d+e+f
  //      (i+1 - (size + 1 - (i + 1)))
  //      (i + 1 - size - 1 +i + 1)
  //      (2*i - size + 1)
  // d: d-a + d-b + d-c + d-d + e-d +f-d
  // d * (1+1+1+1-1-1-1)
  // i=3 2*3-6+1=1
  // soFar = a+b
  // sum = a+b+c+d+e+f
  // i = 2
  // curr = sum - soFar + nums[i] * (2*i - size + 1)
  // 2 3 5
  // sum = 10
  // soFar = 2
  // i=0 10 - 2 + 2 * (2*0-3+1)=10-6=4 xxx
  // 2-2 + 3-2 + 5-2 = 2 * (1-1-1-1) + (3 + 5)
  // 3-2 + 3-3 + 5-3 = 3 * (1+1-1-1) - 2 + (5)
  //                       (2*1-3+1)       (sum-soFar)
  // 5-2 + 5-3 + 5-5 = 5 * (1+1+1-1) -2-3 + (0)
```

#### Approach

Evaluate some examples, then derive the formula.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun getSumAbsoluteDifferences(nums: IntArray): IntArray {
    val sum = nums.sum()
    var soFar = 0
    return IntArray(nums.size) { i ->
      soFar += nums[i]
      (sum - 2 * soFar + nums[i] * (2 * i - nums.size + 2))
    }
  }

```

