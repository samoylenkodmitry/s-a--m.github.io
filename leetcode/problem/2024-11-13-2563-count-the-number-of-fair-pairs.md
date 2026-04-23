---
layout: leetcode-entry
title: "2563. Count the Number of Fair Pairs"
permalink: "/leetcode/problem/2024-11-13-2563-count-the-number-of-fair-pairs/"
leetcode_ui: true
entry_slug: "2024-11-13-2563-count-the-number-of-fair-pairs"
---

[2563. Count the Number of Fair Pairs](https://leetcode.com/problems/count-the-number-of-fair-pairs/description/) medium
[blog post](https://leetcode.com/problems/count-the-number-of-fair-pairs/solutions/6040302/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13112024-2563-count-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/BFBDfbwV5ws)
[deep-dive](https://notebooklm.google.com/notebook/0243e1e2-00c8-45aa-acdf-bc8a78b0a468/audio)
![1.webp](/assets/leetcode_daily_images/67dde262.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/799

#### Problem TLDR

Count pairs `a[i] + a[j] in lower..upper` #medium #binary_search #two_pointers

#### Intuition

I`ve failed this.
First, don't fall into a trick: `order doesn't matter`.
Next, for each number we can do a binary search for its lower and upper bound (Rust solution).

Another optimization: `lower and upper bound only decrease`, we don't have to do a BinarySearch, just decrease the pointers (Kotlin solution).

Another way of thinking of this problem: count two-sum lower than upper, and subtract count two-sum lower than lower (C++ solution).

#### Approach

* Kotlin's binarySearch can return any position of duplicates, so lower_bound must be handwritten
* Rust's partition_point is good
* sometimes problem description is intentionally misleading

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countFairPairs(nums: IntArray, lower: Int, upper: Int): Long {
        nums.sort(); var res = 0L
        var from = nums.size; var to = from
        for ((i, n) in nums.withIndex()) {
            while (from > i + 1 && nums[from - 1] + n >= lower) from--
            while (to > from && nums[to - 1] + n > upper) to--
            res += max(0, to - max(i + 1, from))
        }
        return res
    }

```
```rust

    pub fn count_fair_pairs(mut nums: Vec<i32>, lower: i32, upper: i32) -> i64 {
        nums.sort_unstable();
        let mut res = 0i64;
        for (i, &n) in nums.iter().enumerate() {
            let from = nums.partition_point(|&x| x < lower - n).max(i + 1) as i64;
            let to = nums.partition_point(|&x| x <= upper - n).max(i + 1) as i64;
            res += 0.max(to - from)
        }
        res
    }

```
```c++

    long long countFairPairs(vector<int>& a, int l, int u) {
        sort(begin(a), end(a));
        long long r = 0;
        for (int i = 0, j = a.size() - 1; i < j; r += j - i++)
            while (i < j && a[i] + a[j] > u) --j;
        for (int i = 0, j = a.size() - 1; i < j; r -= j - i++)
            while (i < j && a[i] + a[j] > l - 1) --j;
        return r;
    }

```

