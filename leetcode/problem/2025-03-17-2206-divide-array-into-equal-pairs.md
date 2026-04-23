---
layout: leetcode-entry
title: "2206. Divide Array Into Equal Pairs"
permalink: "/leetcode/problem/2025-03-17-2206-divide-array-into-equal-pairs/"
leetcode_ui: true
entry_slug: "2025-03-17-2206-divide-array-into-equal-pairs"
---

[2206. Divide Array Into Equal Pairs](https://leetcode.com/problems/divide-array-into-equal-pairs/description/) easy
[blog post](https://leetcode.com/problems/divide-array-into-equal-pairs/solutions/6546102/kotlin-rust-by-samoylenkodmitry-oaw2/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17032025-2206-divide-array-into-equal?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_vIbeZRnzjY)
![1.webp](/assets/leetcode_daily_images/6a70484f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/930

#### Problem TLDR

All numbers in pairs #easy #counting #sorting

#### Intuition

Many ways:
* HashMap
* counting, we have at most 500 elements each at most 500
* sorting, can help to save memory
* BitSet, we only have to track parity bit

#### Approach

* implement all
* we can round-wrap bitset into a single 64-bit value, as test-cases are passing right now (but can fail for two bits overlapping from different numbers)

#### Complexity

- Time complexity:
$$O(n)$$, or O(nlog(n)) for sort

- Space complexity:
$$O(n)$$, or O(1) for sort

#### Code

```kotlin

    fun divideArray(nums: IntArray) = nums
        .sorted().chunked(2).all { it[0] == it[1] }

```
``kotlin

    fun divideArray(nums: IntArray) = nums
        .groupBy { it }.all { it.value.size % 2 < 1 }

```
```rust

    pub fn divide_array(mut n: Vec<i32>) -> bool {
        n.sort();
        n.chunk_by(|a, b| a == b).all(|c| c.len() % 2 < 1)
    }

```
```rust

    pub fn divide_array(nums: Vec<i32>) -> bool {
        let mut f = vec![0; 501];
        for x in nums { f[x as usize] ^= 1; f[0] += f[x as usize] * 2 - 1 }
        f[0] < 1
    }

```
```c++

    bool divideArray(vector<int>& n) {
        int f[501];
        for (int x: n) *f += (f[x] ^= 1) * 2 - 1;
        return !*f;
    }

```
```c++

    bool divideArray(vector<int>& n) {
        bitset<501> f;
        for (int x: n) f[x] = !f[x];
        return !f.any();
    }

```
```c++

    bool divideArray(vector<int>& n) {
        long long f = 0;
        for (int x: n) f ^= 1LL << (x % 64);
        return !f;
    }

```

