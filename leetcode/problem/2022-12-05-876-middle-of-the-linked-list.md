---
layout: leetcode-entry
title: "876. Middle of the Linked List"
permalink: "/leetcode/problem/2022-12-05-876-middle-of-the-linked-list/"
leetcode_ui: true
entry_slug: "2022-12-05-876-middle-of-the-linked-list"
---

[876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/) easy

[https://t.me/leetcode_daily_unstoppable/42](https://t.me/leetcode_daily_unstoppable/42)

```kotlin

  fun middleNode(head: ListNode?, fast: ListNode? = head): ListNode? =
        if (fast?.next == null) head else middleNode(head?.next, fast?.next?.next)

```

* one-liner, but in the interview (or production) I would prefer to write a loop

Space: O(n), Time: O(n)

