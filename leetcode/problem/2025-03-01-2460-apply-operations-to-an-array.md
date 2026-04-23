---
layout: leetcode-entry
title: "2460. Apply Operations to an Array"
permalink: "/leetcode/problem/2025-03-01-2460-apply-operations-to-an-array/"
leetcode_ui: true
entry_slug: "2025-03-01-2460-apply-operations-to-an-array"
---

[2460. Apply Operations to an Array](https://leetcode.com/problems/apply-operations-to-an-array/description/) easy
[blog post](https://leetcode.com/problems/apply-operations-to-an-array/solutions/6480393/kotlin-rust-by-samoylenkodmitry-7gbx/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01032025-2460-apply-operations-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/DWsuHRQBX3Y)
![1.webp](/assets/leetcode_daily_images/5ae4264b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/911

#### Problem TLDR

If a[i] == a[i + 1], a[i] *= 2, a[i + 1] = 0 #easy

#### Intuition

The operations should be applied left-to-right. I can't find a way to do this with iterators.

#### Approach

* careful with zeroing if i == j

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun applyOperations(nums: IntArray) = nums.apply {
        for (i in 0..<size - 1) if (nums[i] == nums[i + 1])
            nums[i + 1] = 0.also { nums[i] *= 2 }
    }.sortedBy { it == 0 }

```
```rust

    pub fn apply_operations(mut nums: Vec<i32>) -> Vec<i32> {
        let mut j = 0;
        for i in 0..nums.len() {
            if i < nums.len() - 1 && nums[i] == nums[i + 1]
                { nums[i + 1] = 0; nums[i] *= 2 }
            if nums[i] > 0 { nums.swap(i, j); j += 1 }
        }; nums
    }

```
```c++

    vector<int> applyOperations(vector<int>& a) {
        for (int i = 0; i < size(a) - 1; ++i)
            if (a[i] == a[i + 1]) a[i] *= 2, a[i + 1] = 0;
        stable_partition(begin(a), end(a), [](int n){return n;});
        return a;
    }

```

