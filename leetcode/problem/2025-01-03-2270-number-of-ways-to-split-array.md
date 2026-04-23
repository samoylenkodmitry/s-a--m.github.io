---
layout: leetcode-entry
title: "2270. Number of Ways to Split Array"
permalink: "/leetcode/problem/2025-01-03-2270-number-of-ways-to-split-array/"
leetcode_ui: true
entry_slug: "2025-01-03-2270-number-of-ways-to-split-array"
---

[2270. Number of Ways to Split Array](https://leetcode.com/problems/number-of-ways-to-split-array/description/) medium
[blog post](https://leetcode.com/problems/number-of-ways-to-split-array/solutions/6223552/kotlin-rust-by-samoylenkodmitry-rpz3/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03012025-2270-number-of-ways-to-split?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/MjsYNhMxHnM)
[deep-dive](https://notebooklm.google.com/notebook/5eed8070-7e6d-4ca7-9616-9e1f9c30dd63/audio)
![1.webp](/assets/leetcode_daily_images/65a48ca8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/853

#### Problem TLDR

Count splits left_sum >= right_sum #medium #prefix_sum

#### Intuition

Prefix sum can help solve this.

#### Approach

* careful with an `int` overflow
* this is not about the balance and con't be done in a single pass, as adding negative number decreases the sum, we should hold `left` and `right` part separately

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun waysToSplitArray(nums: IntArray): Int {
        var r = nums.sumOf { it.toLong() }; var l = 0L
        return (0..<nums.lastIndex).count {
            l += nums[it]; r -= nums[it]; l >= r
        }
    }

```

```rust

    pub fn ways_to_split_array(nums: Vec<i32>) -> i32 {
        let (mut l, mut r) = (0, nums.iter().map(|&x| x as i64).sum());
        (0..nums.len() - 1).filter(|&i| {
            l += nums[i] as i64; r -= nums[i] as i64; l >= r
        }).count() as _
    }

```

```c++

    int waysToSplitArray(vector<int>& nums) {
        int res = 0; long long r = reduce(begin(nums), end(nums), 0LL), l = 0;
        for (int i = 0; i < nums.size() - 1; ++i)
            res += (l += nums[i]) >= (r -= nums[i]);
        return res;
    }

```

