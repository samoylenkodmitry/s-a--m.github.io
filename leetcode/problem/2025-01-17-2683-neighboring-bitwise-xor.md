---
layout: leetcode-entry
title: "2683. Neighboring Bitwise XOR"
permalink: "/leetcode/problem/2025-01-17-2683-neighboring-bitwise-xor/"
leetcode_ui: true
entry_slug: "2025-01-17-2683-neighboring-bitwise-xor"
---

[2683. Neighboring Bitwise XOR](https://leetcode.com/problems/neighboring-bitwise-xor/description/) medium
[blog post](https://leetcode.com/problems/neighboring-bitwise-xor/solutions/6293069/kotlin-rust-by-samoylenkodmitry-l3ow/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17012025-2683-neighboring-bitwise?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/MW183bsRh48)
[deep-dive](https://notebooklm.google.com/notebook/612dfb9a-ae14-4bc4-a40e-2e6b6a3e6c75/audio)
![1.webp](/assets/leetcode_daily_images/5035c656.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/867

#### Problem TLDR

Can restore next-sibl-xored array? #medium #xor

#### Intuition

Observe an example:

```j

    // a b c
    // 1 1 0
    // a^b      a != b      a=1    b=1^1 = 0
    //   b^c    b != c      b=0    c=1^0 = 1
    //     c^a  c == a      c=1    a=0^1 = 1 correct

```

We can assume the initial value `a` and after all-`xor` operation compare if it is the same.

#### Approach

* initial value can be `0` or `1`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun doesValidArrayExist(derived: IntArray) =
        derived.reduce(Int::xor) < 1

```
```rust

    pub fn does_valid_array_exist(derived: Vec<i32>) -> bool {
        derived.into_iter().reduce(|a, b| a ^ b).unwrap() < 1
    }

```
```c++

    bool doesValidArrayExist(vector<int>& derived) {
        int a = 1; for (int x: derived) a ^= x;
        return a;
    }

```

