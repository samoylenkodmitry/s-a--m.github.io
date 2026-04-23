---
layout: leetcode-entry
title: "326. Power of Three"
permalink: "/leetcode/problem/2025-08-13-326-power-of-three/"
leetcode_ui: true
entry_slug: "2025-08-13-326-power-of-three"
---

[326. Power of Three](https://leetcode.com/problems/power-of-three/description) easy
[blog post](https://leetcode.com/problems/power-of-three/solutions/7074724/kotlin-rust-by-samoylenkodmitry-7yj7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13082025-326-power-of-three?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/lkY0FNrzSLg)
![1.webp](/assets/leetcode_daily_images/71dbadf0.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1079

#### Problem TLDR

Is n power of 3? #easy #math

#### Intuition

Almost failed at corner case Int.MAX_VALUE, be aware of Int overflow when `*3`.
Another interesting facts:
* 3-base representation has only single `1`, others `0`
* `3^19` is max fit into Int, if `3^19 % n == 0`, n is a power of `3`

#### Approach

* try to write all the different ways
* the `/` solution is the most robust
* some arithmetic fact: `log_3(n) = log_x(n)/log_x(3) for any x`

#### Complexity

- Time complexity:
$$O(log(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 27ms
    fun isPowerOfThree(n: Int) =
        n.toString(3).replace("0", "") == "1"

```
```kotlin

// 10ms
    fun isPowerOfThree(n: Int) = n > 0 &&
        3.0.pow(1.0*log(1.0*n, 3.0).roundToInt()) == 1.0*n

```
```kotlin

// 9ms
    fun isPowerOfThree(n: Int) =
        n > 0 && 3.0.pow(19.0).toInt() % n < 1

```
```kotlin

// 8ms
    fun isPowerOfThree(n: Int): Boolean {
        if (n <= 0) return false
        var x = 1L; val n = 1L * n
        while (x < n) x *= 3
        return x == n
    }

```
```kotlin

// 8ms
    fun isPowerOfThree(n: Int): Boolean {
        var n = n; if (n > 1) while (n % 3 == 0) n /= 3
        return n == 1
    }

```

```rust

// 0ms
    pub fn is_power_of_three(n: i32) -> bool {
        n > 0 && 3i32.pow(19) % n < 1
    }

```
```c++

// 3ms
    bool isPowerOfThree(int n) {
       while (n > 1 && n % 3 == 0) n /= 3;
       return n == 1;
    }

```
```python

// 6ms
    def isPowerOfThree(self, n: int) -> bool:
        return n > 0 and 3**19%n<1

```

