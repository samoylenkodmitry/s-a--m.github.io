---
layout: leetcode-entry
title: "2176. Count Equal and Divisible Pairs in an Array"
permalink: "/leetcode/problem/2025-04-17-2176-count-equal-and-divisible-pairs-in-an-array/"
leetcode_ui: true
entry_slug: "2025-04-17-2176-count-equal-and-divisible-pairs-in-an-array"
---

[2176. Count Equal and Divisible Pairs in an Array](https://leetcode.com/problems/count-equal-and-divisible-pairs-in-an-array/description/) easy
[blog post](https://leetcode.com/problems/count-equal-and-divisible-pairs-in-an-array/solutions/6659001/kotlin-rust-by-samoylenkodmitry-mdmk/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11042025-2176-count-equal-and-divisible?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ajWJk4FH_Xs)
![1.webp](/assets/leetcode_daily_images/1372f5bf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/961

#### Problem TLDR

Pairs a[i] == b[j], i * j % k == 0 #easy

#### Intuition

The brute force is accepted.

#### Approach

* the problem has also more optimal solution by using `gcd(i, k) * gcd(j, k) % k == 0` equality

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countPairs(nums: IntArray, k: Int) =
        nums.indices.sumOf { i ->
            (i + 1..<nums.size).count { j ->
                (i * j) % k == 0 && nums[i] == nums[j] }}

```
```rust

    pub fn count_pairs(n: Vec<i32>, k: i32) -> i32 {
        (0..n.len()).map(|i| (i + 1..n.len()).filter(|&j|
        (i * j) as i32 % k < 1 && n[i] == n[j]).count() as i32).sum()
    }

```
```c++

    int countPairs(vector<int>& n, int k) {
        int r = 0;
        for (int i = 0; i < size(n); ++i)
            for (int j = i + 1; j < size(n); ++j)
                r += i * j % k < 1 && n[i] == n[j];
        return r;
    }

```

