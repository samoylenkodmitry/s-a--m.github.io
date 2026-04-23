---
layout: leetcode-entry
title: "141. Linked List Cycle"
permalink: "/leetcode/problem/2024-03-06-141-linked-list-cycle/"
leetcode_ui: true
entry_slug: "2024-03-06-141-linked-list-cycle"
---

[141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/description/) easy
[blog post](https://leetcode.com/problems/linked-list-cycle/solutions/4830993/kotlin-c/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06032024-141-linked-list-cycle?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pMt1JySmI-I)
![image.png](/assets/leetcode_daily_images/f17cdc3a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/530

#### Problem TLDR

Detect cycle #easy

#### Intuition

Use two pointers, fast and slow, they will meet sometime.

#### Approach

No Rust in the templates provided, sorry.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun hasCycle(h: ListNode?, f: ListNode? = h?.next): Boolean =
    f != null && (h == f || hasCycle(h?.next, f?.next?.next))

```
```c++

    bool hasCycle(ListNode *s) {
        auto f = s;
        while (f && f->next) {
            s = s->next; f = f->next->next;
            if (s == f) return true;
        }
        return false;
    }

```

