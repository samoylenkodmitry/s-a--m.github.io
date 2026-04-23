---
layout: leetcode-entry
title: "1920. Build Array from Permutation"
permalink: "/leetcode/problem/2025-05-06-1920-build-array-from-permutation/"
leetcode_ui: true
entry_slug: "2025-05-06-1920-build-array-from-permutation"
---

[1920. Build Array from Permutation](https://leetcode.com/problems/build-array-from-permutation/description/) easy
[blog post](https://leetcode.com/problems/build-array-from-permutation/solutions/6719273/kotlin-rust-by-samoylenkodmitry-uyq5/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06052025-1920-build-array-from-permutation?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/BD3EfOChLAc)
![1.webp](/assets/leetcode_daily_images/670dd37e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/980

#### Problem TLDR

n[n[i]] #easy

#### Intuition

The follow up is more tricky: we have to store the result and preserver the initial values somehow, shift bits or do * and % operations.

#### Approach

* do golf
* do follow-up

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin

// 3ms
    fun buildArray(n: IntArray) = n.map { n[it] }

```
```kotlin

// 2ms
    fun buildArray(n: IntArray): IntArray {
        for (i in n.indices) n[i] += (n[n[i]] and 0xFFFF) shl 16
        for (i in n.indices) n[i] = n[i] shr 16
        return n;
    }

```
```kotlin

// 1ms
    fun buildArray(n: IntArray) = IntArray(n.size) { n[n[i]] }

```
```rust

// 0ms
    pub fn build_array(mut n: Vec<i32>) -> Vec<i32> {
        for i in 0..n.len() { n[i] |= (n[n[i] as usize] & 0xFFFF) << 16 }
        for i in 0..n.len() { n[i] >>= 16 } n
    }

```
```rust

// 0ms
    pub fn build_array(n: Vec<i32>) -> Vec<i32> {
        n.iter().map(|&x| n[x as usize]).collect()
    }

```
```c++

// 0ms
    vector<int> buildArray(vector<int>& n) {
        vector<int> r(size(n));
        for (int i = 0; auto& x: n) r[i++] = n[x];
        return r;
    }

```

