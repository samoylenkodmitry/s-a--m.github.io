---
layout: leetcode-entry
title: "2221. Find Triangular Sum of an Array"
permalink: "/leetcode/problem/2025-09-30-2221-find-triangular-sum-of-an-array/"
leetcode_ui: true
entry_slug: "2025-09-30-2221-find-triangular-sum-of-an-array"
---

[2221. Find Triangular Sum of an Array](https://leetcode.com/problems/find-triangular-sum-of-an-array/description) medium
[blog post](https://leetcode.com/problems/find-triangular-sum-of-an-array/solutions/7235930/kotlin-rust-by-samoylenkodmitry-gzhc/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30092025-2221-find-triangular-sum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/jFJZ6cJOcWU)

![1.webp](/assets/leetcode_daily_images/b9c48220.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1128

#### Problem TLDR

Triangle sum % 10 #medium #simulation

#### Intuition

The problem is small 1000, O(n^2) simulation is accepted.

The O(n) intuition (from Stefan Pochmann):
* each position get repeated Pascal's Triangle times
```j
1
1 1
1 2 1
1 3 3 1
1 4 6 4 1
1 5 10 10 5 1
1 6 15 20 15 6 1
```
Each new row value is a binomial coefficient (https://en.wikipedia.org/wiki/Binomial_coefficient)
`mC(k+1) = mCk *(n-1-k)/(k+1)`
Division by `/(k+1)` can't be safely done with `%10`.

#### Approach

* windows.map = zipWithNext

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 226ms
    fun triangularSum(n: IntArray) = (2..n.size)
    .fold(n.asList()){r,_->r.zipWithNext{a,b->(a+b)%10}}[0]

```
```kotlin
    fun triangularSum(n: IntArray): Int {
        var f = 1.toBigInteger()
        var r = 0.toBigInteger()
        for ((i, x) in n.withIndex()) {
            r = (r + f * x.toBigInteger()).mod(10.toBigInteger())
            f = f * (n.size - 1 - i).toBigInteger() / (i + 1).toBigInteger()
        }
        return r.toInt()
    }

```
```rust

// 27ms
    pub fn triangular_sum(n: Vec<i32>) -> i32 {
        (1..n.len()).fold(n,|r,t|r.into_iter().tuple_windows().map(|(a,b)|(a+b)%10).collect())[0]
    }

```

