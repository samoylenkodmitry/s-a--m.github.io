---
layout: leetcode-entry
title: "3289. The Two Sneaky Numbers of Digitville"
permalink: "/leetcode/problem/2025-10-31-3289-the-two-sneaky-numbers-of-digitville/"
leetcode_ui: true
entry_slug: "2025-10-31-3289-the-two-sneaky-numbers-of-digitville"
---

[3289. The Two Sneaky Numbers of Digitville](https://leetcode.com/problems/the-two-sneaky-numbers-of-digitville/description/) easy
[blog post](https://leetcode.com/problems/the-two-sneaky-numbers-of-digitville/solutions/7315730/kotlin-rust-by-samoylenkodmitry-g6fn/)
[substack]()
[youtube](https://youtu.be/axzoIKlXIT4)

![5cd75984-aef3-440e-932d-e589bbaac92e (1).webp](/assets/leetcode_daily_images/4111ef31.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1159

#### Problem TLDR

Two extra numbers from 0..n #easy

#### Intuition

Use any of:
* HashSet for visited
* bitmask for visited
* array itself for visited

The clever solution with bit manipulation:
* total xor, no extras: a^b^c -- can compute as x1
* total xor for single extra: a^b^c^a
* total xor for two extras: a^b^c^a^b -- can compute as x2
* xor of x1^x2: a^b^c ^ a^b^c^a^b = a^b -- can compute as x1^x2

Now we have a^b, each bits is a different between `a` and `b`.

Split all given numbers by have or have-nots of this bit.
xor(have_bit) = xx1
xor(have_not_bit) == xx2

Then split range numbers similarly:
xor(have_bit) == yy1
xor(have_not_bit) == yy2

Then a = xx1 ^ yy1, b = xx2 ^ yy2

#### Approach

* just brute-force

#### Complexity

- Time complexity:
$$O()$$

- Space complexity:
$$O()$$

#### Code

```kotlin
// 17ms
    fun getSneakyNumbers(n: IntArray) =
        n.indices.filter { x -> n.count { it == x } > 1 }

```
```rust
// 0ms
    pub fn get_sneaky_numbers(n: Vec<i32>) -> Vec<i32> {
        let mut m = 0u128;
        n.into_iter().filter(|x| { let u = (1 << x) & m > 0; m |= 1<<x; u}).collect()
    }

```

