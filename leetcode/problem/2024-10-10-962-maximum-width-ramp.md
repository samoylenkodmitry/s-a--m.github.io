---
layout: leetcode-entry
title: "962. Maximum Width Ramp"
permalink: "/leetcode/problem/2024-10-10-962-maximum-width-ramp/"
leetcode_ui: true
entry_slug: "2024-10-10-962-maximum-width-ramp"
---

[962. Maximum Width Ramp](https://leetcode.com/problems/maximum-width-ramp/description/) medium
[blog post](https://leetcode.com/problems/maximum-width-ramp/solutions/5894777/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10102024-962-maximum-width-ramp?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/sGNhRD894sM)
![1.webp](/assets/leetcode_daily_images/aabe7b7a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/764

#### Problem TLDR

Max `j-i` between `a[i] <= a[j]` in an array #medium #monotonic_stack #sorting

#### Intuition

The simple monotonic stack will *not* solve this: we should not drop the values on *any* increase/decrease.

Let's think what else we can do, *sort*, for example:

```j

    // 3 7 2 4 9 6 8 1 0 5
    // 0 0 1 1 1 4 4 8 9 9
    // * *                 (3, 7) min = 3, max = 7
    //     * * *           (2, 4, 9) min = 2, max = 9
    //           * *       (6, 8) + (2), min=2, max = 9
    //               *     min=2, max=9
    //                 * * min=2, max=9

```
On the sorted order we can track a `min` and `max` index, and reset the `max` when a new `min` happens. This solution is accepted and it is O(nlog(n))

However, there is a monotonic stack solution that exists. This stack should be the `j` indices in a strictly decreasing order and as right as possible.

#### Approach

* try several ways to transform the data, sorting, monotonic stacks, see what is helpful for the problem

#### Complexity

- Time complexity:
$$O(n)$$ or O(nlogn)

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maxWidthRamp(nums: IntArray): Int {
        val inds = nums.indices.sortedBy { nums[it] }
        var min = nums.size; var max = -1
        return inds.maxOf { i ->
            max = if (i < min) i else max(max, i)
            min = min(min, i)
            max - min
        }
    }

```
```rust

    pub fn max_width_ramp(nums: Vec<i32>) -> i32 {
        let (mut stack, mut res) = (vec![], 0);
        stack.push(nums.len() - 1);
        for (i, &n) in nums.iter().enumerate().rev() {
            if n > nums[*stack.last().unwrap()] { stack.push(i) }}
        for (i, &n) in nums.iter().enumerate() {
            while stack.len() > 0 && n <= nums[*stack.last().unwrap()] {
                res = res.max(stack.pop().unwrap() - i) }}
        res as i32
    }

```
```c++

    int maxWidthRamp(vector<int>& n) {
        vector<int> s; int res = 0;
        for (int i = n.size() - 1; i >= 0; --i)
            if (s.empty() || n[i] > n[s.back()]) s.push_back(i);
        for (int i = 0; i < n.size() && !s.empty(); ++i)
            while (!s.empty() && n[i] <= n[s.back()])
                res = max(res, s.back() - i), s.pop_back();
        return res;
    }

```

