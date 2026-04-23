---
layout: leetcode-entry
title: "232. Implement Queue using Stacks"
permalink: "/leetcode/problem/2022-12-16-232-implement-queue-using-stacks/"
leetcode_ui: true
entry_slug: "2022-12-16-232-implement-queue-using-stacks"
---

[232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/description/) easy

[https://t.me/leetcode_daily_unstoppable/53](https://t.me/leetcode_daily_unstoppable/53)

[blog post](https://leetcode.com/problems/implement-queue-using-stacks/solutions/2918693/kotlin-head-tail/)

```kotlin
class MyQueue() {
	val head = Stack<Int>()
	val tail = Stack<Int>()

	//  []       []
	//  1 2 3 4 -> 4 3 2 - 1
	//  5         4 3 2
	//            4 3 2 5
	fun push(x: Int) {
		head.push(x)
	}

	fun pop(): Int {
		peek()

		return tail.pop()
	}

	fun peek(): Int {
		if (tail.isEmpty()) while(head.isNotEmpty()) tail.push(head.pop())

		return tail.peek()
	}

	fun empty(): Boolean = head.isEmpty() && tail.isEmpty()

}

```

One stack for the head of the queue and other for the tail.
When we need to do `pop` we first drain from one stack to another, so items order will be restored.
* we can skip rotation on push if we fill tail only when its empty

Space: O(1), Time: O(1)

