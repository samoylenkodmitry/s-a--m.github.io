---
layout: leetcode-entry
title: "2441. Largest Positive Integer That Exists With Its Negative"
permalink: "/leetcode/problem/2024-05-02-2441-largest-positive-integer-that-exists-with-its-negative/"
leetcode_ui: true
entry_slug: "2024-05-02-2441-largest-positive-integer-that-exists-with-its-negative"
---

[2441. Largest Positive Integer That Exists With Its Negative](https://leetcode.com/problems/largest-positive-integer-that-exists-with-its-negative/description/) easy
[blog post](https://leetcode.com/problems/largest-positive-integer-that-exists-with-its-negative/solutions/5099630/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02052024-2441-largest-positive-integer?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/qbhha9HEXxU)
![2024-05-02_08-34.webp](/assets/leetcode_daily_images/0bec9eb6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/590

#### Problem TLDR

Max number that has its negative in array #easy #two_pointers

#### Intuition

One possible solution is to sort array and compare minimums with maximums by moving two pointers from left and right of the array.
Another way is to remember which numbers are seen and choose the maximum of them.

#### Approach

* For the second solution, we can use just a [2000] array, as the total count is not that big.

#### Complexity

- Time complexity:
$$O(nlog(n))$$ and $$O(n)$$

- Space complexity:
$$O(1)$$ and $$O(n)$$

#### Code

```kotlin

    fun findMaxK(nums: IntArray): Int {
        nums.sort()
        var i = 0; var j = nums.lastIndex
        while (i < j)
            if (nums[i] == -nums[j]) return nums[j]
            else if (-nums[i] < nums[j]) j-- else i++
        return -1
    }

```
```rust

    pub fn find_max_k(nums: Vec<i32>) -> i32 {
        let (mut counts, mut res) = (vec![0; 2001], -1);
        for x in nums {
            if counts[1000 - x as usize] > 0 { res = res.max(x.abs()) }
            counts[x as usize + 1000] += 1
        }; res
    }

```

