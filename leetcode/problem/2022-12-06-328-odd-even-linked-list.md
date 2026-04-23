---
layout: leetcode-entry
title: "328. Odd Even Linked List"
permalink: "/leetcode/problem/2022-12-06-328-odd-even-linked-list/"
leetcode_ui: true
entry_slug: "2022-12-06-328-odd-even-linked-list"
---

[328. Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/description/) medium

[https://t.me/leetcode_daily_unstoppable/43](https://t.me/leetcode_daily_unstoppable/43)

```kotlin

       // 1 2
    fun oddEvenList(head: ListNode?): ListNode? {
       var odd = head //1
       val evenHead = head?.next
       var even = head?.next //2
       while(even!=null) { //2
           val oddNext = odd?.next?.next //null
           val evenNext = even?.next?.next //null
           odd?.next = oddNext // 1->null
           even?.next = evenNext //2->null
           if (oddNext != null) odd = oddNext //
           even = evenNext // null
       }
       odd?.next = evenHead // 1->2
       return head //1->2->null
    }

```

* be careful and store evenHead in a separate variable

Space: O(1), Time: O(n)

