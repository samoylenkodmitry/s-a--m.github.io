---
layout: leetcode-entry
title: "402. Remove K Digits"
permalink: "/leetcode/problem/2024-04-11-402-remove-k-digits/"
leetcode_ui: true
entry_slug: "2024-04-11-402-remove-k-digits"
---

[402. Remove K Digits](https://leetcode.com/problems/remove-k-digits/description/) medium
[blog post](https://leetcode.com/problems/remove-k-digits/solutions/5006557/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11042024-402-remove-k-digits?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/1uNiFjUan0c)
![2024-04-11_09-09.webp](/assets/leetcode_daily_images/b8f5d1ec.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/568

#### Problem TLDR

Minimum number after removing `k` digits #medium

#### Intuition

Let's observe some examples:
```j
    // 1432219    k=3
    // *
    //  *       14
    //   *      13  1, remove 4
    //    *     12  2, remove 3
    //     *    122
    //      *   121 3, remove 2

    // 12321    k=1
    // *        1
    //  *       12
    //   *      123
    //    *     122, remove 3
```
We can use `increasing stack` technique to choose which characters to remove: remove all tail that less than a new added char.

#### Approach

We can use `Stack` or just a `StringBuilder` directly. Counter is optional, but also helps to save one line of code.
* we can skip adding `0` when string is empty

#### Complexity

- Time complexity:
$$O(n)$$, n^2 when using `deletaAt(0)`, but time is almost the same (we can use a separate counter to avoid this)

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun removeKdigits(num: String, k: Int) = buildString {
        for (i in num.indices) {
            while (i - length < k && length > 0 && last() > num[i])
                setLength(lastIndex)
            append(num[i])
        }
        while (num.length - length < k) setLength(lastIndex)
        while (firstOrNull() == '0') deleteAt(0)
    }.takeIf { it.isNotEmpty() } ?: "0"

```
```rust

    pub fn remove_kdigits(num: String, mut k: i32) -> String {
        let mut sb = String::with_capacity(num.len() - k as usize);
        for c in num.chars() {
            while k > 0 && sb.len() > 0 && sb.chars().last().unwrap() > c {
                sb.pop();
                k -= 1
            }
            if !sb.is_empty() || c != '0' { sb.push(c) }
        }
        for _ in 0..k { sb.pop(); }
        if sb.is_empty() { sb.push('0') }
        sb
    }

```

