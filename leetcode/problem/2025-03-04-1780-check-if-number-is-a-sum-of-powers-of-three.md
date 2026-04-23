---
layout: leetcode-entry
title: "1780. Check if Number is a Sum of Powers of Three"
permalink: "/leetcode/problem/2025-03-04-1780-check-if-number-is-a-sum-of-powers-of-three/"
leetcode_ui: true
entry_slug: "2025-03-04-1780-check-if-number-is-a-sum-of-powers-of-three"
---

[1780. Check if Number is a Sum of Powers of Three](https://leetcode.com/problems/check-if-number-is-a-sum-of-powers-of-three/description/) medium
[blog post](https://leetcode.com/problems/check-if-number-is-a-sum-of-powers-of-three/solutions/6494085/kotlin-rust-by-samoylenkodmitry-fjjv/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04032025-1780-check-if-number-is?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/918j9CJpwKg)
![1.webp](/assets/leetcode_daily_images/08053708.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/914

#### Problem TLDR

Number as the sum of distinct powers of 3 #medium #math

#### Intuition

I already familiar with the binary representation trick: 1011 = 2^3 + 0 + 2^1 + 2^0. Let's observe the problem for base 3:

```j

    // 12/3 = 4   12%3 = 0
    // 4/3 = 1    4%3 = 1
    // 1/3 = 0    1%3 = 1

    // 91/3 = 30 91%3 = 1    3^4
    // 30/3 = 10 30%3 = 0    3^3
    // 10/3 = 3  10%3 = 1    3^2
    // 3/3 = 1    3%3 = 0    3^1
    // 1/3 = 0    1%3 = 1    3^0

    // 21/3 =7   21%3 = 0
    //  7/3 = 2   7%3 = 1
    //  2/3 = 0   2%3 = 2 x

```
The `distinct` requirement means no power can have `2` as multiplier. Or, the result in base 3 should only contain `1` or `0`.

Relevant wiki: https://en.wikipedia.org/wiki/Sums_of_powers

#### Approach

* we can manually check %3
* we can use backtracking and just brute-force: take current power or skip, the depth is log3(n)
* we can write a joke golf by converting to string with radix
* we can optimize with the div_mod function

#### Complexity

- Time complexity:
$$O(log(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun checkPowersOfThree(n: Int) = "2" !in n.toString(3)

```
```rust

    pub fn check_powers_of_three(mut n: i32) -> bool {
        while n > 0 { if n % 3 > 1 { return false }; n /= 3 } true
    }

```
```rust(recursion)

    pub fn check_powers_of_three(n: i32) -> bool {
        n == 0 || n % 3 < 2 && Self::check_powers_of_three(n / 3)
    }

```
```rust(asm)

    pub fn check_powers_of_three(n: i32) -> bool {
        let mut n = n as u32;
        fn asm_div_rem(a: u32, b: u32) -> (u32, u32) {
            let mut tmp: u32 = a;
            let mut remainder: u32 = 0;
            unsafe {
                asm!(
                    "div {divisor}",
                    inout("eax") tmp,
                    inout("edx") remainder,
                    divisor = in(reg) b,
                    options(pure, nomem, nostack),
                );
            }
            (tmp, remainder)
        }
        while n > 0 {
            let (x, r) = asm_div_rem(n, 3);
            n = x;
            if r > 1 { return false }
        } true
    }

```
```c++

    bool checkPowersOfThree(int n, int x = 1) {
        return !n || x <= n &&
            (checkPowersOfThree(n, x * 3) || checkPowersOfThree(n - x, x * 3));
    }

```

