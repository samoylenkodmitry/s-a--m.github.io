---
layout: leetcode-entry
title: "3264. Final Array State After K Multiplication Operations I"
permalink: "/leetcode/problem/2024-12-16-3264-final-array-state-after-k-multiplication-operations-i/"
leetcode_ui: true
entry_slug: "2024-12-16-3264-final-array-state-after-k-multiplication-operations-i"
---

[3264. Final Array State After K Multiplication Operations I](https://leetcode.com/problems/final-array-state-after-k-multiplication-operations-i/description/) easy
[blog post](https://leetcode.com/problems/final-array-state-after-k-multiplication-operations-i/solutions/6151467/kotlin-rust-by-samoylenkodmitry-zyup/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16122024-3264-final-array-state-after?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/g3oQ_y-icNo)
[deep-dive](https://notebooklm.google.com/notebook/a1a2ede1-8cc5-4c8b-b017-9f28aaefb553/audio)
![1.webp](/assets/leetcode_daily_images/cf68dd5e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/834

#### Problem TLDR

Mutliply `k` minimums #easy

#### Intuition

The problem size is small, the brute force works.

One improvement is to use a heap.

#### Approach

* will bucket sort work?

#### Complexity

- Time complexity:
$$O(n^2)$$ or nlog(n)

- Space complexity:
$$O(1)$$ or O(n)

#### Code

```kotlin

    fun getFinalState(nums: IntArray, k: Int, multiplier: Int) = nums.apply {
        for (i in 1..k) nums[indexOf(min())] *= multiplier
    }

```
```rust

    pub fn get_final_state(mut nums: Vec<i32>, k: i32, multiplier: i32) -> Vec<i32> {
        let mut h = BinaryHeap::from_iter(nums.iter().enumerate().map(|(i, &x)| (-x, -(i as i32))));
        for i in 0..k {
            let (x, i) = h.pop().unwrap();
            nums[(-i) as usize] *= multiplier;
            h.push((x * multiplier, i));
        }; nums
    }

```
```c++

    vector<int> getFinalState(vector<int>& nums, int k, int multiplier) {
        while (k--) {
            int j = 0;
            for (int i = 0; i < nums.size(); ++i) if (nums[i] < nums[j]) j = i;
            nums[j] *= multiplier;
        } return nums;
    }

```

