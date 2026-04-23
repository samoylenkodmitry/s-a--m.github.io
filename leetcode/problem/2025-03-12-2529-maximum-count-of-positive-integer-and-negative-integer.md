---
layout: leetcode-entry
title: "2529. Maximum Count of Positive Integer and Negative Integer"
permalink: "/leetcode/problem/2025-03-12-2529-maximum-count-of-positive-integer-and-negative-integer/"
leetcode_ui: true
entry_slug: "2025-03-12-2529-maximum-count-of-positive-integer-and-negative-integer"
---

[2529. Maximum Count of Positive Integer and Negative Integer](https://leetcode.com/problems/maximum-count-of-positive-integer-and-negative-integer/description) easy
[blog post](https://leetcode.com/problems/maximum-count-of-positive-integer-and-negative-integer/solutions/6527587/kotlin-rust-by-samoylenkodmitry-l7lj/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12032025-2529-maximum-count-of-positive?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2K9WG_0c5o8)
![1.webp](/assets/leetcode_daily_images/4e2436fb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/925

#### Problem TLDR

Max positive count or negative count #easy #binary_search

#### Intuition

The brute-force is accepted. However, it is interesting to explore built-in solutions in each languages.

#### Approach

* Kotlin: the shortest is a brute force, no built-in for array Binary Search
* Rust: partition_point
* c++: equal_range is the perfect match

#### Complexity

- Time complexity:
$$O(n)$$ or O(log(n))

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maximumCount(nums: IntArray) =
        max(nums.count { it > 0 }, nums.count { it < 0 })

```
```kotlin

    fun maximumCount(nums: IntArray) = max(
        -nums.asList().binarySearch { if (it < 0) -1 else 1 } - 1,
        nums.size + nums.asList().binarySearch { if (it < 1) -1 else 1 } + 1)

```
```rust

    pub fn maximum_count(nums: Vec<i32>) -> i32 {
        let a = nums.partition_point(|&x| x < 0);
        let b = nums[a..].partition_point(|&x| x < 1);
        a.max(nums[a..].len() - b) as i32
    }

```
```c++

    int maximumCount(vector<int>& n) {
        auto [a, b] = equal_range(begin(n), end(n), 0);
        return max(distance(begin(n), a), distance(b, end(n)));
    }

```
```c++

    int maximumCount(vector<int>& nums) {
        int p = 0, n = 0;
        for (int x: nums) p += x > 0, n += x < 0;
        return max(p, n);
    }

```

