---
layout: leetcode-entry
title: "1749. Maximum Absolute Sum of Any Subarray"
permalink: "/leetcode/problem/2025-02-26-1749-maximum-absolute-sum-of-any-subarray/"
leetcode_ui: true
entry_slug: "2025-02-26-1749-maximum-absolute-sum-of-any-subarray"
---

[1749. Maximum Absolute Sum of Any Subarray](https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/description/) medium
[blog post](https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/solutions/6469370/kotlin-rust-by-samoylenkodmitry-ga1v/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26022025-1749-maximum-absolute-sum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ArX_Thtx1Ms)
![1.webp](/assets/leetcode_daily_images/2b1a94e3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/907

#### Problem TLDR

Max abs of subarray sum #medium #prefix_sum

#### Intuition

Let's observe how the prefix sums can help:

```j

    // 2 -5 1 -4 3 -2
    // i                2
    //    i             2-5=-3    [-3-2=5] max positive
    //      i           2-5+1=-2  [-2-2=-4][-2+3=1]
    //         i        2-5+1-4=-6
    //           i      2-5+1-4+3=-3
    //              i   2-5+1-4+3-2=-5

```
At every index we are interested only in `max positive` and `max negative` previous prefix sums.

Interesting observation from u/lee215/ is we don't care in which order the prefixes sums are, just pick any two, or select the `max` and `min` from them.

#### Approach

* we can skip the current prefix sum variable and just use cumulative max and min: `max = max(x, x + max), min = min(x, x + min)` (Rust solution)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maxAbsoluteSum(nums: IntArray): Int {
        var a = 0; var b = 0; var s = 0; var r = 0
        for (x in nums) {
            s += x; r = maxOf(r, a - s, s - b); a = max(a, s); b = min(b, s)
        }
        return r
    }

```
```rust

    pub fn max_absolute_sum(nums: Vec<i32>) -> i32 {
        let (mut a, mut b, mut r) = (0, 0, 0);
        for x in nums {
            a = x.min(a + x); b = x.max(b + x); r = b.max(-a).max(r)
        }; r
    }

```
```c++

    int maxAbsoluteSum(vector<int>& nums) {
        int a = 0, b = 0, s = 0;
        for (int x: nums) s += x, a = min(a, s), b = max(b, s);
        return b - a;
    }

```

