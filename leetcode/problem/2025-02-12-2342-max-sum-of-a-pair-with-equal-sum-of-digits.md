---
layout: leetcode-entry
title: "2342. Max Sum of a Pair With Equal Sum of Digits"
permalink: "/leetcode/problem/2025-02-12-2342-max-sum-of-a-pair-with-equal-sum-of-digits/"
leetcode_ui: true
entry_slug: "2025-02-12-2342-max-sum-of-a-pair-with-equal-sum-of-digits"
---

[2342. Max Sum of a Pair With Equal Sum of Digits](https://leetcode.com/problems/max-sum-of-a-pair-with-equal-sum-of-digits/description/) medium
[blog post](https://leetcode.com/problems/max-sum-of-a-pair-with-equal-sum-of-digits/solutions/6411856/kotlin-rust-by-samoylenkodmitry-nyox/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12022025-2342-max-sum-of-a-pair-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Z1BSOG9Ld3w)
![1.webp](/assets/leetcode_daily_images/337e0ece.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/893

#### Problem TLDR

Max pairs sum with equal digits sum #medium

#### Intuition

Group numbers by digits sums, find two largest elements.

#### Approach

* the maximum key is 9 * 9 = 81
* shortest golf requires sorting, time degrades 300ms vs 14ms

#### Complexity

- Time complexity:
$$O(n)$$, O(nlog(n)) for Kotlin golf

- Space complexity:
$$O(1)$$, O(n) for golf

#### Code

```kotlin

    fun maximumSum(nums: IntArray) = nums
        .groupBy { "$it".sumOf { it - '0' } }.filter { it.value.size > 1 }
        .maxOfOrNull { it.value.sorted().takeLast(2).sum() } ?: -1

```
```rust

    pub fn maximum_sum(nums: Vec<i32>) -> i32 {
        let (mut s, mut r) = (vec![0; 99], -1);
        for x in nums {
            let (mut k, mut n) = (0, x as usize);
            while n > 0 { k += n % 10; n /= 10 }
            if s[k] > 0 { r = r.max(s[k] + x) }; s[k] = s[k].max(x)
        }; r
    }

```
```c++

    int maximumSum(vector<int>& nums) {
        int s[99]{}, r = -1;
        for (int x: nums) {
            int k = 0, n = x; for (;n; n /= 10) k += n % 10;
            r = max(r, s[k] ? s[k] + x : r);
            s[k] = max(s[k], x);
        } return r;
    }

```

