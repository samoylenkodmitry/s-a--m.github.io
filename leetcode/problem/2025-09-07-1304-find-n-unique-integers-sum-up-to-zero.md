---
layout: leetcode-entry
title: "1304. Find N Unique Integers Sum up to Zero"
permalink: "/leetcode/problem/2025-09-07-1304-find-n-unique-integers-sum-up-to-zero/"
leetcode_ui: true
entry_slug: "2025-09-07-1304-find-n-unique-integers-sum-up-to-zero"
---

[1304. Find N Unique Integers Sum up to Zero](https://leetcode.com/problems/find-n-unique-integers-sum-up-to-zero/description) easy
[blog post](https://leetcode.com/problems/find-n-unique-integers-sum-up-to-zero/solutions/7164836/kotlin-rust-by-samoylenkodmitry-p7po/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07092025-1304-find-n-unique-integers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Rg4LSvUK90Y)

![1.webp](/assets/leetcode_daily_images/a4b15489.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1105

#### Problem TLDR

Any n uniq numbers with sum of 0 #easy

#### Intuition

* fill symmetrical -i,i, then remove 0 if n is even
* derive the law `1-n+i*2` (from lee)
* fill range `2..n` then add `-sum` of that

#### Approach

* careful with even/odd

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 9ms
    fun sumZero(n: Int) = (2..n)+-(2..n).sum()

```
```kotlin

// 0ms
    fun sumZero(n: Int) = IntArray(n) {1-n+it*2}

```
```rust

// 0ms
    pub fn sum_zero(n: i32) -> Vec<i32> {
        (1..n).chain([(n-n*n)/2]).collect()
    }

```
```c++

// 0ms
    vector<int> sumZero(int n) {
        vector<int> r(n); iota(begin(r),end(r),1);
        r.back() = (n-n*n)/2; return r;
    }

```
```python

// 0ms
    sumZero = lambda _,n:[*range(1-n,n,2)]

```

