---
layout: leetcode-entry
title: "1800. Maximum Ascending Subarray Sum"
permalink: "/leetcode/problem/2025-02-04-1800-maximum-ascending-subarray-sum/"
leetcode_ui: true
entry_slug: "2025-02-04-1800-maximum-ascending-subarray-sum"
---

[1800. Maximum Ascending Subarray Sum](https://leetcode.com/problems/maximum-ascending-subarray-sum/description/) easy
[blog post](https://leetcode.com/problems/maximum-ascending-subarray-sum/solutions/6372426/kotlin-rust-by-samoylenkodmitry-zdpp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04022025-1800-maximum-ascending-subarray?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/AQyTzVAzSWY)
![1.webp](/assets/leetcode_daily_images/cc3683a4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/885

#### Problem TLDR

Max increasing subarray sum #easy

#### Intuition

Use brute-force, two-pointers or running sum.

#### Approach

* Rust has a nice chunk_by

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maxAscendingSum(nums: IntArray) =
        nums.indices.maxOf { i ->
            var j = i + 1; while (j < nums.size && nums[j] > nums[j - 1]) j++
            nums.slice(i..<j).sum()
        }

```
```rust

    pub fn max_ascending_sum(nums: Vec<i32>) -> i32 {
        nums.chunk_by(|a, b| a < b).map(|c| c.iter().sum()).max().unwrap()
    }

```
```c++

    int maxAscendingSum(vector<int>& n) {
        int r = n[0];
        for (int i = 1, s = n[0]; i < size(n); ++i)
            r = max(r, s = n[i - 1] < n[i] ? s + n[i] : n[i]);
        return r;
    }

```

