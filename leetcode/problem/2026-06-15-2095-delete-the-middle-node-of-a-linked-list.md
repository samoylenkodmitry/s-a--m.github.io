---
layout: leetcode-entry
title: "2095. Delete the Middle Node of a Linked List"
permalink: "/leetcode/problem/2026-06-15-2095-delete-the-middle-node-of-a-linked-list/"
leetcode_ui: true
entry_slug: "2026-06-15-2095-delete-the-middle-node-of-a-linked-list"
---

[2095. Delete the Middle Node of a Linked List](https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/solutions/8335124/kotlin-rust-by-samoylenkodmitry-983u/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15062026-2095-delete-the-middle-node?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_ZUzITaa0vI)

https://dmitrysamoylenko.com/leetcode/

![15.06.2026.webp](/assets/leetcode_daily_images/15.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1391

#### Problem TLDR

Remove the middle of Linked LIst

#### Intuition

* fast & slow pointer
* count, then walk again

#### Approach

* Rust: fast & slow can be done with unsafe + raw pointers

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun deleteMiddle(h: ListNode?) = run {
        val d = ListNode(0).apply{next = h}
        var s: ListNode? = d; var f = h
        while(f?.next!=null){s = s?.next;f=f?.next?.next}
        s?.next = s?.next?.next; d.next
    }
```
```rust
    pub fn delete_middle(mut h: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut cnt = 0; let mut s = &h;
        while let Some(n) = &s { s = &n.next; cnt += 1 }
        if cnt <= 1 { return None } let mut s = &mut h;
        for i in 0..cnt/2-1 { s = &mut s.as_mut().unwrap().next  }
        let mid = s.as_mut().unwrap().next.take();
        s.as_mut().unwrap().next = mid.and_then(|mut n| n.next.take());
        h
    }
```

