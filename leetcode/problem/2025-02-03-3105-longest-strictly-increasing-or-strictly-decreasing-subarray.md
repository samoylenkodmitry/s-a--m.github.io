---
layout: leetcode-entry
title: "3105. Longest Strictly Increasing or Strictly Decreasing Subarray"
permalink: "/leetcode/problem/2025-02-03-3105-longest-strictly-increasing-or-strictly-decreasing-subarray/"
leetcode_ui: true
entry_slug: "2025-02-03-3105-longest-strictly-increasing-or-strictly-decreasing-subarray"
---

[3105. Longest Strictly Increasing or Strictly Decreasing Subarray](https://leetcode.com/problems/longest-strictly-increasing-or-strictly-decreasing-subarray/description/) easy
[blog post](https://leetcode.com/problems/longest-strictly-increasing-or-strictly-decreasing-subarray/solutions/6366396/kotlin-rust-by-samoylenkodmitry-yuyb/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03022025-3105-longest-strictly-increasing?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XfoPWcZhqKQ)
![1.webp](/assets/leetcode_daily_images/9d9cca08.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/884

#### Problem TLDR

Longest strict monotonic subarray #easy

#### Intuition

Don't forget we can use brute force when the problem size is small. Sometimes that code can be easy to write and check.

#### Approach

* the optimal solution is not that different from the brute force: drop the counter to 1

#### Complexity

- Time complexity:
$$O(n)$$, O(n^2) for the brute-force

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun longestMonotonicSubarray(nums: IntArray) =
        nums.indices.maxOf { i ->
            var a = i + 1; var b = a
            while (a < nums.size && nums[a - 1] > nums[a]) a++
            while (b < nums.size && nums[b - 1] < nums[b]) b++
            max(b - i, a - i) }

```
```rust

    pub fn longest_monotonic_subarray(nums: Vec<i32>) -> i32 {
        nums.chunk_by(|a, b| a > b).chain(nums.chunk_by(|a, b| a < b))
        .map(|c| c.len() as _).max().unwrap_or(1)
    }

```
```c++

    int longestMonotonicSubarray(vector<int>& n) {
        int a = 1, b = 1, r = 1;
        for (int i = 1; i < size(n); ++i)
            r = max({r, n[i] > n[i - 1] ? ++a : (a = 1),
                        n[i] < n[i - 1] ? ++b : (b = 1)});
        return r;
    }

```

