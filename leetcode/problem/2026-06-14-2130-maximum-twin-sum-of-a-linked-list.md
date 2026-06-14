---
layout: leetcode-entry
title: "2130. Maximum Twin Sum of a Linked List"
permalink: "/leetcode/problem/2026-06-14-2130-maximum-twin-sum-of-a-linked-list/"
leetcode_ui: true
entry_slug: "2026-06-14-2130-maximum-twin-sum-of-a-linked-list"
---

[2130. Maximum Twin Sum of a Linked List](https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/solutions/8333049/kotlin-rust-by-samoylenkodmitry-hnu0/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14062026-2130-maximum-twin-sum-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/218om3UJuZI)

https://dmitrysamoylenko.com/leetcode/

![14.06.2026.webp](/assets/leetcode_daily_images/14.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1390

#### Problem TLDR

Max sum pairwise from tail

#### Intuition

Put into a list or revert the first half.

#### Approach

* Kotlin: generateSequence
* Rust: from_fn

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n|1)$$

#### Code

```kotlin
    fun pairSum(h: ListNode?) =
        generateSequence(h){it.next}.map{it.`val`}
        .toList().run {zip(reversed(), Int::plus).max()}
```
```rust
    pub fn pair_sum(mut h: Option<Box<ListNode>>) -> i32 {
       let l: Vec<_> = from_fn(||h.take().map(|n|{h=n.next; n.val})).collect();
       l.iter().rev().zip(&l).map(|(a,b)|a+b).max().unwrap()
    }
```

