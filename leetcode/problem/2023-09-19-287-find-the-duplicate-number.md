---
layout: leetcode-entry
title: "287. Find the Duplicate Number"
permalink: "/leetcode/problem/2023-09-19-287-find-the-duplicate-number/"
leetcode_ui: true
entry_slug: "2023-09-19-287-find-the-duplicate-number"
---

[287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/description/) medium
[blog post](https://leetcode.com/problems/find-the-duplicate-number/solutions/4062911/kotlin-modify-then-revert-42222-also-the-case/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19092023-287-find-the-duplicate-number?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/010d4c9f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/344

#### Problem TLDR

Found duplicate in array, each value is in `1..<arr.size`

#### Intuition
Hint: `4 2 2 2 2 ... 2 ` is also the case.
What we can see, is that every value is in the `1..<arr.size` range, so we can temporarly store the flag in here, then revert it back in the end.

```
    //   0 1 2 3 4  sz = 5
    //   3 1 3 4 2
    // 3       *
    // 1   *
    // 3       x
    //
```

#### Approach
For a flag we can just add some big value to the number, or make it negative, for example.

Let's write it using some Kotlin's API:
* first
* also - notice how it doesn't require brackets

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun findDuplicate(nums: IntArray) = nums.first { n ->
        nums[n % nums.size] >= nums.size
        .also { nums[n % nums.size] += nums.size }
      } % nums.size
      .also { for (j in nums.indices) nums[j] %= nums.size }

```

