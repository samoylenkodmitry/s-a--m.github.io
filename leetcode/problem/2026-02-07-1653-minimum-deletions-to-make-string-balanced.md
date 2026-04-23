---
layout: leetcode-entry
title: "1653. Minimum Deletions to Make String Balanced"
permalink: "/leetcode/problem/2026-02-07-1653-minimum-deletions-to-make-string-balanced/"
leetcode_ui: true
entry_slug: "2026-02-07-1653-minimum-deletions-to-make-string-balanced"
---

[1653. Minimum Deletions to Make String Balanced](https://leetcode.com/problems/minimum-deletions-to-make-string-balanced/description) medium
[blog post](https://leetcode.com/problems/minimum-deletions-to-make-string-balanced/solutions/7560273/kotlin-rust-by-samoylenkodmitry-dhxr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07022026-1653-minimum-deletions-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/tm9vYGMMZMI)

![f761f76d-8f1c-40ff-83bb-b0d5ee98e67a (1).webp](/assets/leetcode_daily_images/7746762f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1261

#### Problem TLDR

Min removal to balance 'ab' #medium #greedy

#### Intuition

Split at each position and compare left and right counts of 'a' and 'b'.
The greedy intuition: count 'b', at each 'a' choose min of (remove 'a'=d+1, keep 'a'=b count removed)

#### Approach

* the corner cases of single 'a' and 'b' should be checked

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 24ms
    fun minimumDeletions(s: String): Int {
        var b = 0
        return min(0,s.minOf { b += 2*(it - 'a')-1; b }) + s.count{it=='a'}
    }
```
```rust
// 10ms
    pub fn minimum_deletions(s: String) -> i32 {
        s.bytes().fold((0,0),|(b,d),c|if c>b'a'{(b+1,d)}else{(b,b.min(d+1))}).1
    }
```

