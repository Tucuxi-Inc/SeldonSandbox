export function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-gray-500">
      <p className="text-lg">{message}</p>
    </div>
  );
}
