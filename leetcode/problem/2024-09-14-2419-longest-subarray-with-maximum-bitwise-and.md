---
layout: leetcode-entry
title: "2419. Longest Subarray With Maximum Bitwise AND"
permalink: "/leetcode/problem/2024-09-14-2419-longest-subarray-with-maximum-bitwise-and/"
leetcode_ui: true
entry_slug: "2024-09-14-2419-longest-subarray-with-maximum-bitwise-and"
---

[2419. Longest Subarray With Maximum Bitwise AND](https://leetcode.com/problems/longest-subarray-with-maximum-bitwise-and/description/) medium
[blog post](https://leetcode.com/problems/longest-subarray-with-maximum-bitwise-and/solutions/5784642/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14092024-2419-longest-subarray-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_jzxy9V-q5o)
![1.webp](/assets/leetcode_daily_images/5f4362a6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/735

#### Problem TLDR

Max bitwise `AND` subarray #medium #bit_manipulation #two_pointers

#### Intuition

Let's observe the problem:

```j

    // 1  001
    // 2  010  [1 2]=000
    // 3  011  [1 2 3]
    // 4  100

```

After some time, the intuition comes: if we have a `maximum` value, every other value would decrease it with `AND` operation.
So, we should only care about the maximum and find the longest subarray of it.

#### Approach

* we can find a `max`, then scan the array, or do this in one go
* we can use indexes and compute `i - j + 1` (i and j must be inclusive)
* or we can use a counter (it is somewhat simpler)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun longestSubarray(nums: IntArray): Int {
        val maxValue = nums.max(); var count = 0
        return nums.maxOf {
            if (it < maxValue) count = 0 else count++
            count
        }
    }

```
```rust

    pub fn longest_subarray(nums: Vec<i32>) -> i32 {
        let (mut j, mut max, mut max_v) = (0, 0, 0);
        for (i, &n) in nums.iter().enumerate() {
            if n > max_v { max_v = n; max = 0; j = i }
            else if n < max_v { j = i + 1 }
            max = max.max(i - j + 1)
        }; max as _
    }

```
```c++

    int longestSubarray(vector<int>& nums) {
        int count, max, max_v = 0;
        for (auto n: nums)
            if (n > max_v) { max_v = n; count = 1; max = 1; }
            else if (n < max_v) count = 0;
            else max = std::max(max, ++count);
        return max;
    }

```

