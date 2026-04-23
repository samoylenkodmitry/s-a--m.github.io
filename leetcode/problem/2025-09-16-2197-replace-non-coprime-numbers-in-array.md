---
layout: leetcode-entry
title: "2197. Replace Non-Coprime Numbers in Array"
permalink: "/leetcode/problem/2025-09-16-2197-replace-non-coprime-numbers-in-array/"
leetcode_ui: true
entry_slug: "2025-09-16-2197-replace-non-coprime-numbers-in-array"
---

[2197. Replace Non-Coprime Numbers in Array](https://leetcode.com/problems/replace-non-coprime-numbers-in-array/description/) hard
[blog post](https://leetcode.com/problems/replace-non-coprime-numbers-in-array/solutions/7195633/kotlin-rust-by-samoylenkodmitry-tvnk/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16092025-2197-replace-non-coprime?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/eTH2hFqhVKY)

![1.webp](/assets/leetcode_daily_images/089de5cd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1114

#### Problem TLDR

Simulate gcd to lcm adjacent pair removal #hard #simulation

#### Intuition

Solved with the hint: only update values to the left.

As all the ways lead to the same result, that means we can pick the more comfortable way: just scan from the left to the right.

GCD(a, b) = (b, a%b)
LCM(a, b) = a*b/GCD(a,b)

#### Approach

* careful with int overflow

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 64ms
    fun replaceNonCoprimes(nums: IntArray) = buildList {
        fun gcd(a: Int, b: Int): Int = if (a % b == 0) b else gcd(b, a % b)
        fun lcm(a: Int, b: Int): Int = ((1L * a * b) / gcd(a, b)).toInt()
        for (x in nums) {
            this += x
            while (size > 1 && gcd(last(), this[size-2]) > 1)
                this += lcm(removeLast(), removeLast())
        }
    }

```
```rust

// 18ms
    pub fn replace_non_coprimes(nums: Vec<i32>) -> Vec<i32> {
        fn gcd(a: i32, b: i32) -> i32 { if a%b > 0 { gcd(b, a%b)} else { b }};
        let lcm = |a: i32, b: i32| (a as i64 * b as i64/ gcd(a, b) as i64) as i32;
        let mut res = vec![];
        for x in nums { res.push(x); while res.len() > 1 && gcd(res[res.len()-1], res[res.len()-2]) > 1 {
            let (a, b) = (res.pop().unwrap(), res.pop().unwrap()); res.push(lcm(a, b))
        }} res
    }

```

