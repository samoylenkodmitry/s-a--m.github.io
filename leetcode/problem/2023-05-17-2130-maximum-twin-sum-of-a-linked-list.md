---
layout: leetcode-entry
title: "2130. Maximum Twin Sum of a Linked List"
permalink: "/leetcode/problem/2023-05-17-2130-maximum-twin-sum-of-a-linked-list/"
leetcode_ui: true
entry_slug: "2023-05-17-2130-maximum-twin-sum-of-a-linked-list"
---

[2130. Maximum Twin Sum of a Linked List](https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/description/) medium
[blog post](https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/solutions/3532758/kotlin-stack/)
[substack](https://dmitriisamoilenko.substack.com/p/17052023-2130-maximum-twin-sum-of?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/215
#### Problem TLDR
Max sum of head-tail twin ListNodes: `a-b-c-d -> max(a+d, b+c)`
#### Intuition
Add first half to the `Stack`, then pop until end reached.
#### Approach
* use `fast` and `slow` pointers to find the center.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

        fun pairSum(head: ListNode?): Int {
            var fast = head
            var slow = head
            var sum = 0
            val stack = Stack<Int>()
                while (fast != null) {
                    stack.add(slow!!.`val`)
                    slow = slow.next
                    fast = fast.next?.next
                }
                while (slow != null) {
                    sum = maxOf(sum, stack.pop() + slow.`val`)
                    slow = slow.next
                }
                return sum
            }

```

