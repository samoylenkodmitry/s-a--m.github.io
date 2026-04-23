---
layout: leetcode-entry
title: "1653. Minimum Deletions to Make String Balanced"
permalink: "/leetcode/problem/2024-07-30-1653-minimum-deletions-to-make-string-balanced/"
leetcode_ui: true
entry_slug: "2024-07-30-1653-minimum-deletions-to-make-string-balanced"
---

[1653. Minimum Deletions to Make String Balanced](https://leetcode.com/problems/minimum-deletions-to-make-string-balanced/description/) medium
[blog post](https://leetcode.com/problems/minimum-deletions-to-make-string-balanced/solutions/5556762/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30072024-1653-minimum-deletions-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7yJlzVZqSG0)
![2024-07-30_08-26_1.webp](/assets/leetcode_daily_images/cf4a9533.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/686

#### Problem TLDR

Min removals to sort 'ab' string #medium

#### Intuition

Let's try every `i` position and count how many `b` are on the left and how many `a` on the right side.

Another solution is a clever one: we count every `b` that is left to the `a` and remove it. For situations like `bba` where we should remove `a` this also works, as we remove `one position` of the incorrect order.

#### Approach

Let's implement first solution in Kotlin and second in Rust.
* as we count `bl` at the current position, we should consider corner case of `countA` removals

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minimumDeletions(s: String): Int {
        val countA = s.count { it == 'a' }; var bl = 0
        return min(countA, s.indices.minOf {
            if (s[it] == 'b') bl++
            bl + (countA - it - 1 + bl)
        })
    }

```
```rust

    pub fn minimum_deletions(s: String) -> i32 {
        let (mut bl, mut del) = (0, 0);
        for b in s.bytes() {
            if b == b'b' { bl += 1 }
            else if bl > 0 { del += 1; bl -= 1 }
        }; del
    }

```

