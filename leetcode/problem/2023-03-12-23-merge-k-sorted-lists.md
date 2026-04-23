---
layout: leetcode-entry
title: "23. Merge k Sorted Lists"
permalink: "/leetcode/problem/2023-03-12-23-merge-k-sorted-lists/"
leetcode_ui: true
entry_slug: "2023-03-12-23-merge-k-sorted-lists"
---

[23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/description/) hard

[blog post](https://leetcode.com/problems/merge-k-sorted-lists/solutions/3287757/kotlin-pq-and-divide-and-conquer/)

```kotlin

    fun mergeKLists(lists: Array<ListNode?>): ListNode? {
        val root = ListNode(0)
        var curr: ListNode = root
        val pq = PriorityQueue<ListNode>(compareBy( { it.`val` }))
        lists.forEach { if (it != null) pq.add(it) }
        while (pq.isNotEmpty()) {
            val next = pq.poll()
            curr.next = next
            next.next?.let { pq.add(it) }
            curr = next
        }
        return root.next
    }
    fun mergeKLists2(lists: Array<ListNode?>): ListNode? {
        fun merge(oneNode: ListNode?, twoNode: ListNode?): ListNode? {
            val root = ListNode(0)
            var curr: ListNode = root
            var one = oneNode
            var two = twoNode
            while (one != null && two != null) {
                if (one.`val` <= two.`val`) {
                    curr.next = one
                    one = one.next
                } else {
                    curr.next = two
                    two = two.next
                }
                curr = curr.next!!
            }
            if (one != null) curr.next = one
            else if (two != null) curr.next = two

            return root.next
        }
        return lists.fold(null as ListNode?) { r, t -> merge(r, t) }
    }

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/146
#### Intuition
On each step, we need to choose a minimum from `k` variables. The best way to do this is to use `PriorityQeueu`
Another solution is to just iteratively merge the `result` to the next list from the array.

#### Approach
* use dummy head
For the `PriorityQueue` solution:
* use non-null values to more robust code
For the iterative solution:
* we can skip merging if one of the lists is empty
#### Complexity
- Time complexity:
* `PriorityQueue`: $$O(nlog(k))$$
* iterative merge: $$O(nk)$$
- Space complexity:
* `PriorityQueue`: $$O(k)$$
* iterative merge: $$O(1)$$

